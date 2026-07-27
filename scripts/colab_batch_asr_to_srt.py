from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".m4a"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ASR videos/audios to SRT with faster-whisper.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad-filter", action="store_true", default=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(output_dir / "batch_asr_to_srt.log")

    files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No supported video/audio files found in {input_dir}")

    logger.info("Found %s input file(s): %s", len(files), ", ".join(path.name for path in files))
    from faster_whisper import WhisperModel

    load_started = time.monotonic()
    logger.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s (first run can spend minutes downloading weights)",
        args.model,
        args.device,
        args.compute_type,
    )
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    logger.info("Model loaded in %.1fs", time.monotonic() - load_started)

    for index, media_path in enumerate(files, start=1):
        duration = _probe_duration(media_path)
        size_mb = media_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Transcribing %s/%s: %s (%.1f MB%s)",
            index,
            len(files),
            media_path,
            size_mb,
            f", {duration:.1f}s" if duration is not None else "",
        )
        transcribe_started = time.monotonic()
        segments_iter, info = model.transcribe(
            str(media_path),
            language=args.language or None,
            vad_filter=args.vad_filter,
            beam_size=args.beam_size,
        )
        logger.info("Decoder returned segment iterator for %s; consuming segments...", media_path.name)
        srt_segments = []
        for seg in segments_iter:
            text = " ".join(seg.text.strip().split())
            if not text:
                continue
            srt_segments.append({"start": float(seg.start), "end": float(seg.end), "text": text})
            if len(srt_segments) == 1 or len(srt_segments) % 25 == 0:
                logger.info(
                    "Progress %s: %s subtitle segment(s), audio position %.1fs",
                    media_path.name,
                    len(srt_segments),
                    float(seg.end),
                )

        output_path = output_dir / f"{_safe_stem(media_path)}.srt"
        output_path.write_text(_to_srt(srt_segments), encoding="utf-8")
        logger.info(
            "Wrote %s segment(s) to %s detected_language=%s probability=%.4f",
            len(srt_segments),
            output_path,
            getattr(info, "language", None),
            getattr(info, "language_probability", 0.0),
        )
        logger.info("Finished %s in %.1fs", media_path.name, time.monotonic() - transcribe_started)

    zip_base = output_dir.parent / "asr_srt_results"
    if zip_base.with_suffix(".zip").exists():
        zip_base.with_suffix(".zip").unlink()
    shutil.make_archive(str(zip_base), "zip", output_dir)
    logger.info("Created zip: %s.zip", zip_base)


def _probe_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return stem or "media"


def _to_srt(segments: list[dict]) -> str:
    blocks = []
    for index, seg in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{_srt_timestamp(seg['start'])} --> {_srt_timestamp(seg['end'])}\n"
            f"{seg['text']}\n"
        )
    return "\n".join(blocks)


def _srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("dichvideo_batch_asr_to_srt")
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
