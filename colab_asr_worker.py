from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="DichVideo Colab ASR worker")
    parser.add_argument("--job-dir", required=True, help="Path to one DichVideo job folder in Google Drive.")
    parser.add_argument("--model", default=None, help="Override Whisper model from manifest.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"], help="ASR device.")
    parser.add_argument("--compute-type", default="float16", help="faster-whisper compute type.")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    log_path = job_dir / "logs" / "colab_asr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(log_path)

    try:
        manifest_path = job_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("config", {})
        chunks = manifest.get("chunks", [])
        model_name = args.model or config.get("whisper_model") or "small"
        language = config.get("source_language")

        _write_status(job_dir, "running_asr", f"Colab worker started with model={model_name}.")
        logger.info("Job dir: %s", job_dir)
        logger.info("Chunk count: %s", len(chunks))
        logger.info("Loading faster-whisper model=%s device=%s compute_type=%s language=%s", model_name, args.device, args.compute_type, language or "auto")

        from faster_whisper import WhisperModel

        model = WhisperModel(model_name, device=args.device, compute_type=args.compute_type)
        all_segments = []
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        for chunk in chunks:
            chunk_index = chunk["index"]
            chunk_path = job_dir / chunk["path"]
            offset = float(chunk.get("offset", 0.0))
            logger.info("Transcribing chunk=%s offset=%.3f path=%s", chunk_index, offset, chunk_path)
            _write_status(job_dir, "running_asr", f"Transcribing chunk {chunk_index + 1}/{len(chunks)}")

            segments, info = model.transcribe(
                str(chunk_path),
                language=language,
                vad_filter=True,
                beam_size=5,
            )
            chunk_segments = []
            logger.info(
                "Chunk=%s detected_language=%s probability=%.4f",
                chunk_index,
                getattr(info, "language", None),
                getattr(info, "language_probability", 0.0),
            )
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                item = {
                    "id": len(all_segments),
                    "start": round(float(seg.start) + offset, 3),
                    "end": round(float(seg.end) + offset, 3),
                    "text": text,
                    "language": getattr(info, "language", None),
                }
                all_segments.append(item)
                chunk_segments.append(item)

            _write_json(output_dir / f"transcript_chunk_{chunk_index:04d}.json", chunk_segments)
            logger.info("Chunk=%s segment_count=%s total_segments=%s", chunk_index, len(chunk_segments), len(all_segments))

        _write_json(output_dir / "transcript.json", all_segments)
        _write_status(job_dir, "done_asr", f"ASR done. Segments: {len(all_segments)}")
        logger.info("ASR done. Total segments: %s", len(all_segments))
    except Exception as exc:
        logger.exception("Colab ASR worker failed")
        _write_status(job_dir, "error", str(exc))
        raise


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("dichvideo_colab_asr")
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


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_status(job_dir: Path, status: str, message: str) -> None:
    _write_json(
        job_dir / "status.json",
        {
            "status": status,
            "message": message,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


if __name__ == "__main__":
    main()
