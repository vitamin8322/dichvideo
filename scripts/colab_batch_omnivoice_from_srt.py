from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch OmniVoice TTS from uploaded SRT files.")
    parser.add_argument("--srt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--reference-start", type=float, default=0.0)
    parser.add_argument("--reference-duration", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--trim-start-seconds", type=float, default=0.17)
    parser.add_argument("--num-step", type=int, default=32)
    args = parser.parse_args()

    srt_dir = Path(args.srt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(output_dir / "batch_omnivoice_from_srt.log")
    _check_binary("ffmpeg")
    _check_binary("ffprobe")

    srt_files = sorted(srt_dir.glob("*.srt"))
    if not srt_files:
        raise RuntimeError(f"No .srt files found in {srt_dir}")

    ref_audio = _prepare_reference_audio(
        Path(args.ref_audio),
        output_dir / "reference_24k.wav",
        args.reference_start,
        args.reference_duration,
        logger,
    )

    import soundfile as sf
    import torch
    from omnivoice import OmniVoice

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    logger.info("Loading OmniVoice model=%s device=%s dtype=%s", args.model, args.device, args.dtype)
    model = OmniVoice.from_pretrained(args.model, device_map=args.device, dtype=dtype)

    for srt_index, srt_path in enumerate(srt_files, start=1):
        logger.info("Processing SRT %s/%s: %s", srt_index, len(srt_files), srt_path)
        segments = parse_srt(srt_path.read_text(encoding="utf-8-sig"))
        if not segments:
            logger.warning("Skipping empty SRT: %s", srt_path)
            continue
        _process_one_srt(
            model=model,
            sf=sf,
            srt_path=srt_path,
            segments=segments,
            ref_audio=ref_audio,
            ref_text=args.ref_text,
            output_dir=output_dir,
            speed=args.speed,
            trim_start_seconds=args.trim_start_seconds,
            num_step=args.num_step,
            logger=logger,
        )

    zip_base = output_dir.parent / "omnivoice_audio_results"
    if zip_base.with_suffix(".zip").exists():
        zip_base.with_suffix(".zip").unlink()
    shutil.make_archive(str(zip_base), "zip", output_dir)
    logger.info("Created zip: %s.zip", zip_base)


def _process_one_srt(model, sf, srt_path: Path, segments: list[dict], ref_audio: Path, ref_text: str | None, output_dir: Path, speed: float, trim_start_seconds: float, num_step: int, logger: logging.Logger) -> None:
    name = _safe_stem(srt_path)
    per_srt_dir = output_dir / name
    raw_dir = per_srt_dir / "segments"
    raw_dir.mkdir(parents=True, exist_ok=True)

    scheduled = []
    cursor = 0.0
    for seg in segments:
        text = " ".join(seg["text"].split())
        raw_wav = raw_dir / f"{seg['index']:04d}.wav"
        logger.info("OmniVoice generate srt=%s segment=%s chars=%s", srt_path.name, seg["index"], len(text))
        audio = model.generate(
            text=text,
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            speed=speed,
            num_step=num_step,
        )
        audio_data = _trim_start(audio[0], sample_rate=24000, trim_seconds=trim_start_seconds)
        sf.write(str(raw_wav), audio_data, 24000, subtype="PCM_24")

        raw_duration = _duration(raw_wav, logger)
        scheduled_duration = raw_duration
        scheduled_start = max(seg["start"], cursor)
        cursor = scheduled_start + scheduled_duration

        scheduled.append({
            **seg,
            "scheduled_start": scheduled_start,
            "scheduled_end": scheduled_start + scheduled_duration,
            "audio_duration": scheduled_duration,
            "audio": str(raw_wav.relative_to(per_srt_dir)),
        })

    full_wav = per_srt_dir / f"{name}_full.wav"
    _mix_scheduled_audio(scheduled, full_wav, logger, per_srt_dir)
    (per_srt_dir / f"{name}.srt").write_text(srt_path.read_text(encoding="utf-8-sig"), encoding="utf-8")
    _write_json(per_srt_dir / "timing_schedule.json", scheduled)
    logger.info("Finished %s -> %s", srt_path.name, full_wav)


def parse_srt(content: str) -> list[dict]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        return []
    blocks = re.split(r"\n\s*\n", content)
    segments = []
    fallback_index = 1
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        timing_line_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_line_index is None:
            continue
        maybe_index = lines[0] if timing_line_index > 0 else str(fallback_index)
        try:
            index = int(re.sub(r"\D+", "", maybe_index) or fallback_index)
        except ValueError:
            index = fallback_index
        timing = lines[timing_line_index]
        start_s, end_s = [part.strip().split()[0] for part in timing.split("-->", 1)]
        text = " ".join(lines[timing_line_index + 1:]).strip()
        if text:
            segments.append({
                "index": index,
                "start": _parse_srt_timestamp(start_s),
                "end": _parse_srt_timestamp(end_s),
                "text": text,
            })
            fallback_index += 1
    return segments


def _parse_srt_timestamp(value: str) -> float:
    match = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", value)
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def _mix_scheduled_audio(scheduled: list[dict], output_path: Path, logger: logging.Logger, root_dir: Path) -> None:
    total_duration = max(float(item["scheduled_end"]) for item in scheduled)
    silence_path = root_dir / "_silence.wav"
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=44100", "-t", f"{total_duration:.3f}", str(silence_path)], logger)
    inputs = ["-i", str(silence_path)]
    filters = []
    mix_inputs = ["[0:a]"]
    for input_index, item in enumerate(scheduled, start=1):
        audio_path = root_dir / item["audio"]
        inputs.extend(["-i", str(audio_path)])
        delay_ms = max(0, int(float(item["scheduled_start"]) * 1000))
        label = f"a{input_index}"
        filters.append(f"[{input_index}:a]adelay={delay_ms}:all=1[{label}]")
        mix_inputs.append(f"[{label}]")
    filter_complex = ";".join(filters + [f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0[out]"])
    _run(["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", "[out]", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s24le", str(output_path)], logger)
    silence_path.unlink(missing_ok=True)


def _prepare_reference_audio(input_path: Path, output_path: Path, start: float, duration: float, logger: logging.Logger) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s24le",
        str(output_path),
    ], logger)
    return output_path


def _trim_start(audio_data, sample_rate: int, trim_seconds: float):
    trim_samples = max(0, int(sample_rate * trim_seconds))
    if trim_samples <= 0:
        return audio_data
    if len(audio_data) <= trim_samples:
        return audio_data
    return audio_data[trim_samples:]


def _duration(path: Path, logger: logging.Logger) -> float:
    completed = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], logger)
    return float(completed.stdout.strip())


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return stem or "srt"


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


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("dichvideo_batch_omnivoice_from_srt")
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
