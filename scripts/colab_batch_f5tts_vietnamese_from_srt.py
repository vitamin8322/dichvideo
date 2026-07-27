from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import time
from importlib.resources import files
from pathlib import Path


DEFAULT_MODEL_REPO = "hynt/F5-TTS-Vietnamese-ViVoice"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch F5-TTS Vietnamese 1000h from uploaded SRT files.")
    parser.add_argument("--srt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timing-mode", default="no_cut_sequential", choices=["fit_segments", "no_cut_sequential"])
    parser.add_argument("--max-tempo", type=float, default=1.35)
    parser.add_argument("--reference-start", type=float, default=0.0)
    parser.add_argument("--reference-duration", type=float, default=12.0)
    parser.add_argument("--auto-ref-asr-model", default="small")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--nfe-step", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--cross-fade-duration", type=float, default=0.15)
    parser.add_argument("--disable-text-normalize", action="store_true")
    parser.add_argument("--merge-all-text", action="store_true", help="Merge SRT text before generating audio.")
    parser.add_argument("--merge-scope", default="per_srt", choices=["per_srt", "all"], help="Use with --merge-all-text.")
    args = parser.parse_args()

    srt_dir = Path(args.srt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(output_dir / "batch_f5tts_vietnamese_from_srt.log")
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
    ref_text = args.ref_text.strip()
    if not ref_text:
        ref_text = _transcribe_reference_audio(ref_audio, args.auto_ref_asr_model, logger)

    engine = F5VietnameseEngine(
        model_repo=args.model_repo,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        ref_audio=ref_audio,
        ref_text=ref_text,
        device=args.device,
        speed=args.speed,
        nfe_step=args.nfe_step,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_sampling_coef,
        cross_fade_duration=args.cross_fade_duration,
        logger=logger,
    )

    if args.merge_all_text and args.merge_scope == "all":
        _process_all_srt_merged(
            engine=engine,
            srt_files=srt_files,
            output_dir=output_dir,
            normalize_text=not args.disable_text_normalize,
            logger=logger,
        )
    else:
        for srt_index, srt_path in enumerate(srt_files, start=1):
            logger.info("Processing SRT %s/%s: %s", srt_index, len(srt_files), srt_path)
            segments = parse_srt(srt_path.read_text(encoding="utf-8-sig"))
            if not segments:
                logger.warning("Skipping empty SRT: %s", srt_path)
                continue
            if args.merge_all_text:
                _process_one_srt_merged(
                    engine=engine,
                    srt_path=srt_path,
                    segments=segments,
                    output_dir=output_dir,
                    normalize_text=not args.disable_text_normalize,
                    logger=logger,
                )
            else:
                _process_one_srt(
                    engine=engine,
                    srt_path=srt_path,
                    segments=segments,
                    output_dir=output_dir,
                    timing_mode=args.timing_mode,
                    max_tempo=args.max_tempo,
                    normalize_text=not args.disable_text_normalize,
                    logger=logger,
                )

    zip_base = output_dir.parent / "f5tts_vietnamese_audio_results"
    if zip_base.with_suffix(".zip").exists():
        zip_base.with_suffix(".zip").unlink()
    shutil.make_archive(str(zip_base), "zip", output_dir)
    logger.info("Created zip: %s.zip", zip_base)


def _process_all_srt_merged(
    engine: F5VietnameseEngine,
    srt_files: list[Path],
    output_dir: Path,
    normalize_text: bool,
    logger: logging.Logger,
) -> None:
    merged_dir = output_dir / "merged_all"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_parts = []
    source_segments = []

    for srt_index, srt_path in enumerate(srt_files, start=1):
        logger.info("Processing SRT %s/%s: %s", srt_index, len(srt_files), srt_path)
        segments = parse_srt(srt_path.read_text(encoding="utf-8-sig"))
        if not segments:
            logger.warning("Skipping empty SRT: %s", srt_path)
            continue
        shutil.copy2(srt_path, merged_dir / srt_path.name)
        for seg in segments:
            raw_text = " ".join(seg["text"].split())
            text = normalize_tts_text(raw_text) if normalize_text else raw_text
            merged_parts.append(text)
            source_segments.append({
                "source_srt": srt_path.name,
                "index": seg["index"],
                "start": seg["start"],
                "end": seg["end"],
                "text": raw_text,
                "normalized_text": text,
            })

    if not merged_parts:
        raise RuntimeError("No text found in uploaded SRT files.")
    merged_text = " ".join(merged_parts)
    output_wav = merged_dir / "merged_all_full.wav"
    logger.info("F5-TTS merged-all generate segments=%s chars=%s", len(source_segments), len(merged_text))
    engine.synthesize(merged_text, output_wav)
    audio_duration = _duration(output_wav, logger)
    (merged_dir / "merged_text.txt").write_text(merged_text, encoding="utf-8")
    _write_json(merged_dir / "source_segments.json", source_segments)
    _write_json(merged_dir / "timing_schedule.json", [{
        "id": 0,
        "scheduled_start": 0.0,
        "scheduled_end": audio_duration,
        "audio_duration": audio_duration,
        "audio_path": output_wav.name,
        "text": merged_text,
        "source_segment_count": len(source_segments),
    }])
    logger.info("Finished merged all SRT text -> %s", output_wav)


class F5VietnameseEngine:
    def __init__(
        self,
        model_repo: str,
        model_dir: Path | None,
        ref_audio: Path,
        ref_text: str,
        device: str,
        speed: float,
        nfe_step: int,
        cfg_strength: float,
        sway_sampling_coef: float,
        cross_fade_duration: float,
        logger: logging.Logger,
    ) -> None:
        import torch
        from hydra.utils import get_class
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf
        from f5_tts.infer.utils_infer import (
            infer_process,
            load_model,
            load_vocoder,
            preprocess_ref_audio_text,
        )

        self.infer_process = infer_process
        self.speed = speed
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef
        self.cross_fade_duration = cross_fade_duration
        self.logger = logger

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model_dir is None:
            logger.info("Downloading model repo=%s", model_repo)
            model_dir = Path(snapshot_download(repo_id=model_repo))
        ckpt_file = model_dir / "model_last.pt"
        vocab_file = model_dir / "vocab.txt"
        config_as_vocab = model_dir / "config.json"
        if not vocab_file.exists() and config_as_vocab.exists():
            shutil.copy2(config_as_vocab, vocab_file)
        if not ckpt_file.exists():
            raise RuntimeError(f"Missing F5 checkpoint: {ckpt_file}")
        if not vocab_file.exists():
            raise RuntimeError(f"Missing F5 vocab.txt/config.json in: {model_dir}")

        logger.info("Loading F5-TTS Vietnamese model from %s on %s", model_dir, device)
        vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_Base.yaml")))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        self.model = load_model(
            model_cls,
            model_cfg.model.arch,
            str(ckpt_file),
            mel_spec_type="vocos",
            vocab_file=str(vocab_file),
            device=device,
        )
        self.vocoder = vocoder
        self.ref_audio, self.ref_text = preprocess_ref_audio_text(str(ref_audio), ref_text or "", show_info=logger.info)

    def synthesize(self, text: str, output_path: Path) -> None:
        import soundfile as sf

        clean_text = " ".join(text.split()).strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text")
        self.logger.info("F5-TTS generate chars=%s output=%s text=%s", len(clean_text), output_path, clean_text)
        audio, sample_rate, _ = self.infer_process(
            self.ref_audio,
            self.ref_text,
            clean_text,
            self.model,
            self.vocoder,
            mel_spec_type="vocos",
            target_rms=0.1,
            cross_fade_duration=self.cross_fade_duration,
            nfe_step=self.nfe_step,
            cfg_strength=self.cfg_strength,
            sway_sampling_coef=self.sway_sampling_coef,
            speed=self.speed,
            device=self.device,
        )
        if audio is None:
            raise RuntimeError(f"F5-TTS generated no audio for text: {clean_text}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sample_rate)


def _process_one_srt(
    engine: F5VietnameseEngine,
    srt_path: Path,
    segments: list[dict],
    output_dir: Path,
    timing_mode: str,
    max_tempo: float,
    normalize_text: bool,
    logger: logging.Logger,
) -> None:
    name = _safe_stem(srt_path)
    per_srt_dir = output_dir / name
    raw_dir = per_srt_dir / "segments"
    mix_dir = per_srt_dir / "mix_segments"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mix_dir.mkdir(parents=True, exist_ok=True)

    scheduled = []
    cursor = 0.0
    for seg in segments:
        raw_text = " ".join(seg["text"].split())
        text = normalize_tts_text(raw_text) if normalize_text else raw_text
        raw_wav = raw_dir / f"{seg['index']:04d}.wav"
        mix_wav = mix_dir / f"{seg['index']:04d}.wav"
        logger.info("F5-TTS segment srt=%s segment=%s raw=%s normalized=%s", srt_path.name, seg["index"], raw_text, text)
        engine.synthesize(text, raw_wav)

        original_duration = max(0.1, seg["end"] - seg["start"])
        raw_duration = _duration(raw_wav, logger)
        if timing_mode == "fit_segments":
            _fit_audio(raw_wav, mix_wav, original_duration, max_tempo, trim=True, logger=logger)
            scheduled_start = seg["start"]
            scheduled_duration = original_duration
        else:
            tempo = raw_duration / original_duration if original_duration > 0 else 1.0
            if tempo > 1.0:
                _tempo_audio(raw_wav, mix_wav, min(tempo, max_tempo), logger)
            else:
                _convert_audio(raw_wav, mix_wav, logger)
            scheduled_duration = _duration(mix_wav, logger)
            scheduled_start = max(seg["start"], cursor)
            cursor = scheduled_start + scheduled_duration

        scheduled.append({
            **seg,
            "normalized_text": text,
            "scheduled_start": scheduled_start,
            "scheduled_end": scheduled_start + scheduled_duration,
            "audio_duration": scheduled_duration,
            "raw_audio": str(raw_wav.relative_to(per_srt_dir)),
            "mix_audio": str(mix_wav.relative_to(per_srt_dir)),
        })

    full_wav = per_srt_dir / f"{name}_full.wav"
    _mix_scheduled_audio(scheduled, full_wav, logger, per_srt_dir)
    (per_srt_dir / f"{name}.srt").write_text(srt_path.read_text(encoding="utf-8-sig"), encoding="utf-8")
    _write_json(per_srt_dir / "timing_schedule.json", scheduled)
    logger.info("Finished %s -> %s", srt_path.name, full_wav)


def _process_one_srt_merged(
    engine: F5VietnameseEngine,
    srt_path: Path,
    segments: list[dict],
    output_dir: Path,
    normalize_text: bool,
    logger: logging.Logger,
) -> None:
    name = _safe_stem(srt_path)
    per_srt_dir = output_dir / name
    per_srt_dir.mkdir(parents=True, exist_ok=True)

    merged_parts = []
    source_segments = []
    for seg in segments:
        raw_text = " ".join(seg["text"].split())
        text = normalize_tts_text(raw_text) if normalize_text else raw_text
        merged_parts.append(text)
        source_segments.append({
            "index": seg["index"],
            "start": seg["start"],
            "end": seg["end"],
            "text": raw_text,
            "normalized_text": text,
        })

    if not merged_parts:
        logger.warning("Skipping empty SRT after parsing: %s", srt_path)
        return

    merged_text = " ".join(merged_parts)
    full_wav = per_srt_dir / f"{name}_full.wav"
    logger.info("F5-TTS merged SRT generate srt=%s segments=%s chars=%s", srt_path.name, len(source_segments), len(merged_text))
    engine.synthesize(merged_text, full_wav)
    audio_duration = _duration(full_wav, logger)
    (per_srt_dir / f"{name}.srt").write_text(srt_path.read_text(encoding="utf-8-sig"), encoding="utf-8")
    (per_srt_dir / "merged_text.txt").write_text(merged_text, encoding="utf-8")
    _write_json(per_srt_dir / "source_segments.json", source_segments)
    _write_json(per_srt_dir / "timing_schedule.json", [{
        "id": 0,
        "scheduled_start": 0.0,
        "scheduled_end": audio_duration,
        "audio_duration": audio_duration,
        "audio_path": full_wav.name,
        "text": merged_text,
        "source_segment_count": len(source_segments),
    }])
    logger.info("Finished merged %s -> %s", srt_path.name, full_wav)


LETTER_NAMES = {
    "A": "a",
    "B": "bê",
    "C": "xê",
    "D": "đê",
    "E": "e",
    "F": "ép",
    "G": "gờ",
    "H": "hát",
    "I": "i",
    "J": "giây",
    "K": "ca",
    "L": "eo",
    "M": "em",
    "N": "en",
    "O": "o",
    "P": "pê",
    "Q": "quy",
    "R": "a rờ",
    "S": "ét",
    "T": "tê",
    "U": "u",
    "V": "vê",
    "W": "vê kép",
    "X": "ích",
    "Y": "y",
    "Z": "dét",
}


def normalize_tts_text(text: str) -> str:
    text = " ".join(text.split())

    def rank_repl(match: re.Match) -> str:
        prefix = match.group(1)
        letters = match.group(2)
        spoken = " ".join(LETTER_NAMES.get(ch.upper(), ch.lower()) for ch in letters)
        return f"{prefix} {spoken}"

    text = re.sub(r"\b(cấp|cap|rank|hạng|hang|class)\s+([A-Z]{1,4})\b", rank_repl, text, flags=re.IGNORECASE)

    def acronym_repl(match: re.Match) -> str:
        token = match.group(0)
        if len(token) == 1 and token not in LETTER_NAMES:
            return token
        return " ".join(LETTER_NAMES.get(ch, ch.lower()) for ch in token)

    text = re.sub(r"(?<![A-Za-zÀ-ỹ])([A-Z]{1,6})(?![A-Za-zÀ-ỹ])", acronym_repl, text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


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
        index = int(re.sub(r"\D+", "", maybe_index) or fallback_index)
        start_s, end_s = [part.strip().split()[0] for part in lines[timing_line_index].split("-->", 1)]
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
        audio_path = root_dir / item["mix_audio"]
        inputs.extend(["-i", str(audio_path)])
        delay_ms = max(0, int(float(item["scheduled_start"]) * 1000))
        label = f"a{input_index}"
        filters.append(f"[{input_index}:a]adelay={delay_ms}:all=1[{label}]")
        mix_inputs.append(f"[{label}]")
    filter_complex = ";".join(filters + [f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0[out]"])
    _run(["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", "[out]", "-ac", "2", "-ar", "44100", str(output_path)], logger)
    silence_path.unlink(missing_ok=True)


def _prepare_reference_audio(input_path: Path, output_path: Path, start: float, duration: float, logger: logging.Logger) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", "24000",
        str(output_path),
    ], logger)
    return output_path


def _transcribe_reference_audio(ref_audio: Path, model_name: str, logger: logging.Logger) -> str:
    logger.info("No --ref-text provided. Transcribing reference audio with faster-whisper model=%s", model_name)
    try:
        import torch
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Missing faster-whisper. Install it or pass --ref-text manually.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(str(ref_audio), language="vi", vad_filter=True)
    ref_text = " ".join(segment.text.strip() for segment in segments).strip()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not ref_text:
        raise RuntimeError("Could not transcribe reference audio. Pass --ref-text manually.")
    logger.info("Reference text: %s", ref_text)
    return ref_text


def _fit_audio(input_path: Path, output_path: Path, target_duration: float, max_tempo: float, trim: bool, logger: logging.Logger) -> None:
    source_duration = _duration(input_path, logger)
    tempo = source_duration / target_duration if target_duration > 0 else 1.0
    audio_filter = f"atempo={min(tempo, max_tempo):.5f},apad" if tempo > 1.0 else "apad"
    if trim:
        audio_filter += f",atrim=0:{target_duration:.3f}"
    _run(["ffmpeg", "-y", "-i", str(input_path), "-filter:a", audio_filter, "-ac", "1", "-ar", "44100", str(output_path)], logger)


def _tempo_audio(input_path: Path, output_path: Path, tempo: float, logger: logging.Logger) -> None:
    _run(["ffmpeg", "-y", "-i", str(input_path), "-filter:a", f"atempo={tempo:.5f}", "-ac", "1", "-ar", "44100", str(output_path)], logger)


def _convert_audio(input_path: Path, output_path: Path, logger: logging.Logger) -> None:
    _run(["ffmpeg", "-y", "-i", str(input_path), "-ac", "1", "-ar", "44100", str(output_path)], logger)


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
    logger = logging.getLogger("dichvideo_batch_f5tts_vietnamese_from_srt")
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
