from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="DichVideo Colab OmniVoice TTS worker")
    parser.add_argument("--job-dir", required=True, help="Path to one DichVideo job folder.")
    parser.add_argument("--ref-audio", required=True, help="Reference voice audio, e.g. audio-truyen.mp3.")
    parser.add_argument("--ref-text", default=None, help="Optional transcription of reference audio.")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--timing-mode", default="no_cut_sequential", choices=["fit_segments", "no_cut_sequential"])
    parser.add_argument("--max-tempo", type=float, default=1.35)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--num-step", type=int, default=16)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    log_path = job_dir / "logs" / "omnivoice_tts.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(log_path)

    try:
        _check_binary("ffmpeg")
        _check_binary("ffprobe")
        translated_path = job_dir / "output" / "translated.json"
        if not translated_path.exists():
            raise RuntimeError(f"Missing translated.json: {translated_path}")

        segments = json.loads(translated_path.read_text(encoding="utf-8"))
        ref_audio = _prepare_reference_audio(Path(args.ref_audio), job_dir / "work" / "omnivoice_ref.wav", logger)
        _write_status(job_dir, "running_tts", f"Loading OmniVoice model={args.model}")

        import soundfile as sf
        import torch
        from omnivoice import OmniVoice

        dtype = torch.float16 if args.dtype == "float16" else torch.float32
        logger.info("Loading OmniVoice model=%s device=%s dtype=%s", args.model, args.device, args.dtype)
        model = OmniVoice.from_pretrained(args.model, device_map=args.device, dtype=dtype)

        tts_dir = job_dir / "work" / "omnivoice_tts"
        fit_dir = job_dir / "work" / "fit"
        tts_dir.mkdir(parents=True, exist_ok=True)
        fit_dir.mkdir(parents=True, exist_ok=True)

        if args.timing_mode == "no_cut_sequential":
            scheduled = _generate_sequential(
                model=model,
                sf=sf,
                segments=segments,
                ref_audio=ref_audio,
                ref_text=args.ref_text,
                tts_dir=tts_dir,
                fit_dir=fit_dir,
                max_tempo=args.max_tempo,
                speed=args.speed,
                num_step=args.num_step,
                logger=logger,
                job_dir=job_dir,
            )
        else:
            scheduled = _generate_fit_segments(
                model=model,
                sf=sf,
                segments=segments,
                ref_audio=ref_audio,
                ref_text=args.ref_text,
                tts_dir=tts_dir,
                fit_dir=fit_dir,
                max_tempo=args.max_tempo,
                speed=args.speed,
                num_step=args.num_step,
                logger=logger,
            )

        dubbed_audio = job_dir / "output" / "dubbed.wav"
        _mix_scheduled_audio(scheduled, dubbed_audio, logger, job_dir)
        _write_status(job_dir, "done_tts", f"OmniVoice TTS done. Segments: {len(scheduled)}")
        logger.info("OmniVoice TTS done: %s", dubbed_audio)
    except Exception as exc:
        logger.exception("OmniVoice TTS worker failed")
        _write_status(job_dir, "error", str(exc))
        raise


def _generate_fit_segments(model, sf, segments, ref_audio: Path, ref_text: str | None, tts_dir: Path, fit_dir: Path, max_tempo: float, speed: float, num_step: int, logger: logging.Logger) -> list[dict]:
    scheduled = []
    for seg in segments:
        target_duration = max(0.1, float(seg["end"]) - float(seg["start"]))
        raw_wav = tts_dir / f"seg_{seg['id']:05d}.wav"
        fit_wav = fit_dir / f"seg_{seg['id']:05d}.wav"
        text = seg.get("translated_text") or seg["text"]
        _generate_one(model, sf, text, ref_audio, ref_text, raw_wav, speed, num_step, logger, seg["id"])
        _fit_audio(raw_wav, fit_wav, target_duration, max_tempo, trim=True, logger=logger)
        scheduled.append({**seg, "scheduled_start": float(seg["start"]), "audio_path": fit_wav, "audio_duration": target_duration})
    return scheduled


def _generate_sequential(model, sf, segments, ref_audio: Path, ref_text: str | None, tts_dir: Path, fit_dir: Path, max_tempo: float, speed: float, num_step: int, logger: logging.Logger, job_dir: Path) -> list[dict]:
    scheduled = []
    cursor = 0.0
    for seg in segments:
        original_duration = max(0.1, float(seg["end"]) - float(seg["start"]))
        raw_wav = tts_dir / f"seg_{seg['id']:05d}.wav"
        fit_wav = fit_dir / f"seg_{seg['id']:05d}.wav"
        text = seg.get("translated_text") or seg["text"]
        _write_status(job_dir, "running_tts", f"Generating TTS segment {seg['id'] + 1}/{len(segments)}")
        _generate_one(model, sf, text, ref_audio, ref_text, raw_wav, speed, num_step, logger, seg["id"])
        raw_duration = _duration(raw_wav, logger)
        tempo = raw_duration / original_duration if original_duration > 0 else 1.0
        if tempo > 1.0:
            _tempo_audio(raw_wav, fit_wav, min(tempo, max_tempo), logger)
        else:
            shutil.copy2(raw_wav, fit_wav)
        fitted_duration = _duration(fit_wav, logger)
        scheduled_start = max(float(seg["start"]), cursor)
        scheduled.append({**seg, "scheduled_start": scheduled_start, "audio_path": fit_wav, "audio_duration": fitted_duration})
        cursor = scheduled_start + fitted_duration
        logger.info("Sequential segment id=%s scheduled_start=%.3f duration=%.3f cursor=%.3f", seg["id"], scheduled_start, fitted_duration, cursor)

    _write_json(job_dir / "output" / "timing_schedule.json", [
        {
            "id": item["id"],
            "original_start": item["start"],
            "original_end": item["end"],
            "scheduled_start": round(item["scheduled_start"], 3),
            "audio_duration": round(item["audio_duration"], 3),
            "scheduled_end": round(item["scheduled_start"] + item["audio_duration"], 3),
            "text": item.get("translated_text") or item["text"],
        }
        for item in scheduled
    ])
    return scheduled


def _generate_one(model, sf, text: str, ref_audio: Path, ref_text: str | None, output_path: Path, speed: float, num_step: int, logger: logging.Logger, segment_id: int) -> None:
    clean_text = " ".join(text.split())
    logger.info("OmniVoice generate id=%s chars=%s output=%s", segment_id, len(clean_text), output_path)
    kwargs = {
        "text": clean_text,
        "ref_audio": str(ref_audio),
        "speed": speed,
        "num_step": num_step,
    }
    if ref_text:
        kwargs["ref_text"] = ref_text
    audio = model.generate(**kwargs)
    sf.write(str(output_path), audio[0], 24000)


def _mix_scheduled_audio(scheduled: list[dict], output_path: Path, logger: logging.Logger, job_dir: Path) -> None:
    total_duration = max(float(item["scheduled_start"]) + float(item["audio_duration"]) for item in scheduled)
    silence_path = job_dir / "work" / "silence.wav"
    _run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=44100",
        "-t", f"{total_duration:.3f}", str(silence_path)
    ], logger)

    inputs = ["-i", str(silence_path)]
    filters = []
    mix_inputs = ["[0:a]"]
    for index, item in enumerate(scheduled, start=1):
        inputs.extend(["-i", str(item["audio_path"])])
        delay_ms = max(0, int(float(item["scheduled_start"]) * 1000))
        label = f"a{index}"
        filters.append(f"[{index}:a]adelay={delay_ms}:all=1[{label}]")
        mix_inputs.append(f"[{label}]")

    filter_complex = ";".join(filters + [f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0[out]"])
    _run([
        "ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", "[out]",
        "-ac", "2", "-ar", "44100", str(output_path)
    ], logger)


def _prepare_reference_audio(input_path: Path, output_path: Path, logger: logging.Logger) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", "24000",
        "-t", "10", str(output_path)
    ], logger)
    return output_path


def _fit_audio(input_path: Path, output_path: Path, target_duration: float, max_tempo: float, trim: bool, logger: logging.Logger) -> None:
    source_duration = _duration(input_path, logger)
    tempo = source_duration / target_duration if target_duration > 0 else 1.0
    if tempo > 1.0:
        audio_filter = f"atempo={min(tempo, max_tempo):.5f},apad"
    else:
        audio_filter = "apad"
    if trim:
        audio_filter += f",atrim=0:{target_duration:.3f}"
    _run(["ffmpeg", "-y", "-i", str(input_path), "-filter:a", audio_filter, "-ac", "1", "-ar", "44100", str(output_path)], logger)


def _tempo_audio(input_path: Path, output_path: Path, tempo: float, logger: logging.Logger) -> None:
    _run(["ffmpeg", "-y", "-i", str(input_path), "-filter:a", f"atempo={tempo:.5f}", "-ac", "1", "-ar", "44100", str(output_path)], logger)


def _duration(path: Path, logger: logging.Logger) -> float:
    completed = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], logger)
    return float(completed.stdout.strip())


def _run(cmd: list[str], logger: logging.Logger) -> subprocess.CompletedProcess:
    logger.info("Running command: %s", " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.stdout.strip():
        logger.info("stdout: %s", completed.stdout.strip()[-3000:])
    if completed.stderr.strip():
        logger.info("stderr: %s", completed.stderr.strip()[-3000:])
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(cmd)}")
    return completed


def _check_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing dependency: {name}")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_status(job_dir: Path, status: str, message: str) -> None:
    _write_json(job_dir / "status.json", {"status": status, "message": message, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("dichvideo_omnivoice_tts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


if __name__ == "__main__":
    main()
