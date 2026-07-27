from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import subprocess
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PipelineConfig:
    source_language: str | None = None
    target_language: str = "vi"
    whisper_model: str = "small"
    asr_device: str = "cpu"
    compute_type: str = "int8"
    translation_provider: str = "none"
    openai_model: str = "gpt-4o-mini"
    tts_voice: str = "vi-VN-HoaiMyNeural"
    chunk_minutes: float = 10.0
    max_tempo: float = 1.35
    keep_original_audio: bool = False
    original_audio_volume: float = 0.18
    timing_mode: str = "fit_segments"
    video_when_audio_longer: str = "hold_last_frame"


@dataclass
class PipelineUpdate:
    status: str
    log_text: str
    video_path: str | None = None
    transcript_path: str | None = None
    log_path: str | None = None
    job_path: str | None = None


def run_pipeline(video_path: Path, jobs_root: Path, config: PipelineConfig) -> Iterable[PipelineUpdate]:
    job_dir = _create_job_dir(jobs_root)
    log_path = job_dir / "logs" / "pipeline.log"
    logger = _setup_logger(log_path)
    manifest_path = job_dir / "manifest.json"

    def update(message: str, video: Path | None = None, transcript: Path | None = None) -> PipelineUpdate:
        logger.info(message)
        return PipelineUpdate(
            status=message,
            log_text=_tail(log_path),
            video_path=str(video) if video and video.exists() else None,
            transcript_path=str(transcript) if transcript and transcript.exists() else None,
            log_path=str(log_path),
            job_path=str(job_dir),
        )

    try:
        _check_binary("ffmpeg")
        _check_binary("ffprobe")
        _write_json(manifest_path, {"config": asdict(config), "source_video": str(video_path), "job_dir": str(job_dir)})
        yield update(f"Job created: {job_dir}")

        input_video = job_dir / "input" / video_path.name
        shutil.copy2(video_path, input_video)
        logger.info("Copied source video to %s", input_video)

        audio_path = job_dir / "work" / "source.wav"
        yield update("Extracting audio with ffmpeg...")
        _extract_audio(input_video, audio_path, logger)

        yield update("Splitting audio into chunks...")
        chunks = _split_audio(audio_path, job_dir / "work" / "chunks", config.chunk_minutes, logger)
        logger.info("Chunk count: %s", len(chunks))

        yield update(f"Transcribing {len(chunks)} chunk(s) with faster-whisper...")
        segments = _transcribe_chunks(chunks, config, logger)
        transcript_path = job_dir / "output" / "transcript.json"
        _write_json(transcript_path, segments)
        yield update(f"Transcription done: {len(segments)} segment(s)", transcript=transcript_path)

        yield update(f"Translating segments with provider: {config.translation_provider}")
        translated = _repair_transcript_text(_translate_segments(segments, config, logger), logger)
        translated_path = job_dir / "output" / "translated.json"
        _write_json(translated_path, translated)
        yield update("Translation done", transcript=translated_path)

        yield update("Generating TTS audio and fitting segment timing...")
        dubbed_audio = job_dir / "output" / "dubbed.wav"
        _build_dubbed_audio(translated, dubbed_audio, config, logger, job_dir)

        yield update("Muxing final video...")
        final_video = job_dir / "output" / "final.mp4"
        _mux_video(
            input_video,
            audio_path,
            dubbed_audio,
            final_video,
            config.keep_original_audio,
            config.original_audio_volume,
            config.timing_mode != "no_cut_sequential",
            config.video_when_audio_longer,
            logger,
        )

        yield update("Done", video=final_video, transcript=translated_path)
    except Exception:
        logger.exception("Pipeline failed")
        yield PipelineUpdate(
            status="Pipeline failed. Xem log de biet chi tiet.",
            log_text=_tail(log_path),
            video_path=None,
            transcript_path=None,
            log_path=str(log_path),
            job_path=str(job_dir),
        )


def prepare_colab_asr_job(video_path: Path, remote_jobs_root: Path, config: PipelineConfig) -> Iterable[PipelineUpdate]:
    job_dir = _create_job_dir(remote_jobs_root)
    log_path = job_dir / "logs" / "pipeline.log"
    logger = _setup_logger(log_path)
    manifest_path = job_dir / "manifest.json"

    def update(message: str) -> PipelineUpdate:
        logger.info(message)
        return PipelineUpdate(
            status=message,
            log_text=_tail(log_path),
            video_path=None,
            transcript_path=None,
            log_path=str(log_path),
            job_path=str(job_dir),
        )

    try:
        _check_binary("ffmpeg")
        _check_binary("ffprobe")
        yield update(f"Colab ASR job created: {job_dir}")

        input_video = job_dir / "input" / video_path.name
        shutil.copy2(video_path, input_video)
        logger.info("Copied source video to %s", input_video)

        audio_path = job_dir / "work" / "source.wav"
        yield update("Extracting audio for Colab ASR...")
        _extract_audio(input_video, audio_path, logger)

        yield update("Splitting audio chunks for Colab ASR...")
        chunks = _split_audio(audio_path, job_dir / "work" / "chunks", config.chunk_minutes, logger)
        worker_source = Path(__file__).resolve().parent.parent / "scripts" / "colab_asr_worker.py"
        worker_target = remote_jobs_root.parent / "colab_asr_worker.py"
        if worker_source.exists():
            shutil.copy2(worker_source, worker_target)
            logger.info("Copied Colab worker to %s", worker_target)
        else:
            logger.warning("Colab worker source not found: %s", worker_source)

        manifest = {
            "schema": "dichvideo_colab_asr_v1",
            "status": "pending_asr",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": asdict(config),
            "source_video": str(input_video),
            "job_dir": str(job_dir),
            "chunks": [
                {
                    "index": chunk["index"],
                    "path": _relative_posix(chunk["path"], job_dir),
                    "offset": chunk["offset"],
                }
                for chunk in chunks
            ],
        }
        _write_json(manifest_path, manifest)
        _write_json(job_dir / "status.json", {"status": "pending_asr", "message": "Ready for Colab worker.", "worker_path": str(worker_target)})
        yield update(f"Ready for Colab. Job path: {job_dir}. Worker: {worker_target}")
    except Exception:
        logger.exception("Prepare Colab ASR job failed")
        yield PipelineUpdate(
            status="Prepare Colab ASR job failed. Xem log de biet chi tiet.",
            log_text=_tail(log_path),
            log_path=str(log_path),
            job_path=str(job_dir),
        )


def finish_colab_asr_job(job_dir: Path, config: PipelineConfig) -> Iterable[PipelineUpdate]:
    log_path = job_dir / "logs" / "pipeline.log"
    logger = _setup_logger(log_path)
    manifest_path = job_dir / "manifest.json"
    transcript_path = job_dir / "output" / "transcript.json"

    def update(message: str, video: Path | None = None, transcript: Path | None = None) -> PipelineUpdate:
        logger.info(message)
        return PipelineUpdate(
            status=message,
            log_text=_tail(log_path),
            video_path=str(video) if video and video.exists() else None,
            transcript_path=str(transcript) if transcript and transcript.exists() else None,
            log_path=str(log_path),
            job_path=str(job_dir),
        )

    try:
        if not manifest_path.exists():
            raise RuntimeError(f"Missing manifest.json: {manifest_path}")
        if not transcript_path.exists():
            raise RuntimeError(
                "Chua co transcript tu Colab. Hay mo notebook Colab, chay worker voi JOB_DIR nay, "
                f"doi den khi co file output/transcript.json roi bam hoan tat lai. Missing: {transcript_path}"
            )
        _check_binary("ffmpeg")
        _check_binary("ffprobe")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_video = Path(manifest["source_video"])
        if not input_video.exists():
            local_inputs = sorted((job_dir / "input").glob("*"))
            if not local_inputs:
                raise RuntimeError(f"Source video not found: {input_video}")
            input_video = local_inputs[0]
            logger.info("Manifest source video missing; using local input video: %s", input_video)
        audio_path = job_dir / "work" / "source.wav"
        segments = _repair_transcript_text(json.loads(transcript_path.read_text(encoding="utf-8")), logger)
        logger.info("Loaded Colab transcript: %s segment(s)", len(segments))
        yield update(f"Loaded Colab transcript: {len(segments)} segment(s)", transcript=transcript_path)

        yield update(f"Translating segments with provider: {config.translation_provider}")
        translated = _repair_transcript_text(_translate_segments(segments, config, logger), logger)
        translated_path = job_dir / "output" / "translated.json"
        _write_json(translated_path, translated)
        yield update("Translation done", transcript=translated_path)

        yield update("Generating TTS audio and fitting segment timing...")
        dubbed_audio = job_dir / "output" / "dubbed.wav"
        _build_dubbed_audio(translated, dubbed_audio, config, logger, job_dir)

        yield update("Muxing final video...")
        final_video = job_dir / "output" / "final.mp4"
        _mux_video(
            input_video,
            audio_path,
            dubbed_audio,
            final_video,
            config.keep_original_audio,
            config.original_audio_volume,
            config.timing_mode != "no_cut_sequential",
            config.video_when_audio_longer,
            logger,
        )
        _write_json(job_dir / "status.json", {"status": "done", "message": "Final video ready.", "final_video": str(final_video)})

        yield update("Done", video=final_video, transcript=translated_path)
    except Exception:
        logger.exception("Finish Colab ASR job failed")
        yield PipelineUpdate(
            status="Finish Colab ASR job failed. Xem log de biet chi tiet.",
            log_text=_tail(log_path),
            log_path=str(log_path),
            job_path=str(job_dir),
        )


def translate_colab_asr_job(job_dir: Path, config: PipelineConfig) -> Iterable[PipelineUpdate]:
    log_path = job_dir / "logs" / "pipeline.log"
    logger = _setup_logger(log_path)
    transcript_path = job_dir / "output" / "transcript.json"

    def update(message: str, transcript: Path | None = None) -> PipelineUpdate:
        logger.info(message)
        return PipelineUpdate(
            status=message,
            log_text=_tail_job_logs(job_dir),
            transcript_path=str(transcript) if transcript and transcript.exists() else None,
            log_path=str(log_path),
            job_path=str(job_dir),
        )

    try:
        if not transcript_path.exists():
            raise RuntimeError(
                "Chua co transcript tu Colab. Hay chay ASR worker den khi co output/transcript.json. "
                f"Missing: {transcript_path}"
            )

        segments = _repair_transcript_text(json.loads(transcript_path.read_text(encoding="utf-8")), logger)
        logger.info("Loaded Colab transcript for translation: %s segment(s)", len(segments))
        yield update(f"Loaded transcript: {len(segments)} segment(s)", transcript=transcript_path)

        yield update(f"Translating segments with provider: {config.translation_provider}")
        translated = _repair_transcript_text(_translate_segments(segments, config, logger), logger)
        translated_path = job_dir / "output" / "translated.json"
        _write_json(translated_path, translated)
        _write_json(job_dir / "status.json", {"status": "translated", "message": "translated.json ready.", "translated": str(translated_path)})
        yield update("Translation done. Ready for OmniVoice packaging.", transcript=translated_path)
    except Exception:
        logger.exception("Translate Colab ASR job failed")
        yield PipelineUpdate(
            status="Translate Colab ASR job failed. Xem log de biet chi tiet.",
            log_text=_tail_job_logs(job_dir),
            log_path=str(log_path),
            job_path=str(job_dir),
        )


def package_omnivoice_input_job(job_dir: Path) -> PipelineUpdate:
    log_path = job_dir / "logs" / "pipeline.log"
    logger = _setup_logger(log_path)
    translated_path = job_dir / "output" / "translated.json"
    if not translated_path.exists():
        raise RuntimeError(f"Missing translated.json. Hay dich transcript truoc: {translated_path}")

    zip_path = job_dir.parent / f"{job_dir.name}_omnivoice_input.zip"
    if zip_path.exists():
        zip_path.unlink()

    include_files = [
        job_dir / "manifest.json",
        job_dir / "status.json",
        job_dir / "work" / "source.wav",
        job_dir / "output" / "transcript.json",
        job_dir / "output" / "translated.json",
    ]
    include_dirs = [
        job_dir / "input",
        job_dir / "work" / "chunks",
        job_dir / "logs",
    ]

    logger.info("Packaging OmniVoice input zip: %s", zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in include_files:
            if file_path.exists():
                zf.write(file_path, file_path.relative_to(job_dir.parent))
        for dir_path in include_dirs:
            if dir_path.exists():
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, file_path.relative_to(job_dir.parent))

    logger.info("OmniVoice input package ready: %s size=%s", zip_path, zip_path.stat().st_size)
    return PipelineUpdate(
        status=f"OmniVoice input zip ready: {zip_path}",
        log_text=_tail_job_logs(job_dir),
        transcript_path=str(zip_path),
        log_path=str(log_path),
        job_path=str(job_dir),
    )


def finish_remote_tts_job(job_dir: Path, config: PipelineConfig) -> Iterable[PipelineUpdate]:
    log_path = job_dir / "logs" / "pipeline.log"
    logger = _setup_logger(log_path)
    manifest_path = job_dir / "manifest.json"
    dubbed_audio = job_dir / "output" / "dubbed.wav"
    translated_path = job_dir / "output" / "translated.json"

    def update(message: str, video: Path | None = None, transcript: Path | None = None) -> PipelineUpdate:
        logger.info(message)
        return PipelineUpdate(
            status=message,
            log_text=_tail_job_logs(job_dir),
            video_path=str(video) if video and video.exists() else None,
            transcript_path=str(transcript) if transcript and transcript.exists() else None,
            log_path=str(log_path),
            job_path=str(job_dir),
        )

    try:
        if not manifest_path.exists():
            raise RuntimeError(f"Missing manifest.json: {manifest_path}")
        if not dubbed_audio.exists():
            raise RuntimeError(
                "Chua co audio TTS tu Colab. Hay chay OmniVoice worker den khi co file output/dubbed.wav. "
                f"Missing: {dubbed_audio}"
            )
        _check_binary("ffmpeg")
        _check_binary("ffprobe")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_video = Path(manifest["source_video"])
        if not input_video.exists():
            local_inputs = sorted((job_dir / "input").glob("*"))
            if not local_inputs:
                raise RuntimeError(f"Source video not found: {input_video}")
            input_video = local_inputs[0]
            logger.info("Manifest source video missing; using local input video: %s", input_video)

        source_audio = job_dir / "work" / "source.wav"
        if not source_audio.exists():
            source_audio = input_video

        yield update("Loaded remote TTS dubbed audio", transcript=translated_path if translated_path.exists() else None)
        yield update("Muxing final video from OmniVoice audio...")
        final_video = job_dir / "output" / "final.mp4"
        _mux_video(
            input_video,
            source_audio,
            dubbed_audio,
            final_video,
            config.keep_original_audio,
            config.original_audio_volume,
            config.timing_mode != "no_cut_sequential",
            config.video_when_audio_longer,
            logger,
        )
        _write_json(job_dir / "status.json", {"status": "done", "message": "Final video ready.", "final_video": str(final_video)})
        yield update("Done", video=final_video, transcript=translated_path if translated_path.exists() else None)
    except Exception:
        logger.exception("Finish remote TTS job failed")
        yield PipelineUpdate(
            status="Finish remote TTS job failed. Xem log de biet chi tiet.",
            log_text=_tail_job_logs(job_dir),
            log_path=str(log_path),
            job_path=str(job_dir),
        )


def read_job_status(job_dir: Path) -> PipelineUpdate:
    log_path = job_dir / "logs" / "pipeline.log"
    status_path = job_dir / "status.json"
    transcript_path = job_dir / "output" / "transcript.json"
    final_video = job_dir / "output" / "final.mp4"
    if status_path.exists():
        status_data = json.loads(status_path.read_text(encoding="utf-8"))
        status = f"{status_data.get('status', 'unknown')}: {status_data.get('message', '')}".strip()
    else:
        status = "No status.json found."
    return PipelineUpdate(
        status=status,
        log_text=_tail_job_logs(job_dir),
        video_path=str(final_video) if final_video.exists() else None,
        transcript_path=str(transcript_path) if transcript_path.exists() else None,
        log_path=str(log_path) if log_path.exists() else None,
        job_path=str(job_dir),
    )


def _create_job_dir(jobs_root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    job_dir = jobs_root / f"job_{stamp}"
    for child in ["input", "work", "output", "logs"]:
        (job_dir / child).mkdir(parents=True, exist_ok=True)
    return job_dir


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"dichvideo.{log_path.parent.parent.name}")
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


def _check_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing dependency: {name}. Hay cai ffmpeg va them vao PATH.")


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


def _extract_audio(video_path: Path, audio_path: Path, logger: logging.Logger) -> None:
    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ], logger)


def _split_audio(audio_path: Path, chunks_dir: Path, chunk_minutes: float, logger: logging.Logger) -> list[dict]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    duration = _duration(audio_path, logger)
    chunk_seconds = max(60, int(chunk_minutes * 60))
    chunks = []
    count = max(1, math.ceil(duration / chunk_seconds))
    for index in range(count):
        start = index * chunk_seconds
        out = chunks_dir / f"chunk_{index:04d}.wav"
        _run([
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{chunk_seconds:.3f}",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out),
        ], logger)
        chunks.append({"index": index, "path": out, "offset": float(start)})
    logger.info("Audio duration: %.3fs, chunk seconds: %s", duration, chunk_seconds)
    return chunks


def _duration(path: Path, logger: logging.Logger) -> float:
    completed = _run([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ], logger)
    return float(completed.stdout.strip())


def _transcribe_chunks(chunks: list[dict], config: PipelineConfig, logger: logging.Logger) -> list[dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Missing Python package faster-whisper. Run: py -m pip install -r requirements.txt") from exc

    logger.info(
        "Loading Whisper model=%s device=%s compute_type=%s language=%s",
        config.whisper_model,
        config.asr_device,
        config.compute_type,
        config.source_language or "auto",
    )
    model = WhisperModel(config.whisper_model, device=config.asr_device, compute_type=config.compute_type)
    all_segments = []
    for chunk in chunks:
        logger.info("Transcribing chunk %s offset=%.3f path=%s", chunk["index"], chunk["offset"], chunk["path"])
        segments, info = model.transcribe(
            str(chunk["path"]),
            language=config.source_language,
            vad_filter=True,
            beam_size=5,
        )
        logger.info(
            "Chunk %s detected_language=%s probability=%.4f",
            chunk["index"],
            getattr(info, "language", None),
            getattr(info, "language_probability", 0.0),
        )
        for seg in segments:
            start = float(seg.start) + float(chunk["offset"])
            end = float(seg.end) + float(chunk["offset"])
            text = seg.text.strip()
            if not text:
                continue
            all_segments.append({
                "id": len(all_segments),
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "language": getattr(info, "language", None),
            })
    logger.info("Total transcribed segments: %s", len(all_segments))
    return all_segments


def _translate_segments(segments: list[dict], config: PipelineConfig, logger: logging.Logger) -> list[dict]:
    if config.translation_provider == "none":
        return [{**seg, "translated_text": seg["text"]} for seg in segments]
    if config.translation_provider == "deep-translator":
        return _translate_deep_translator(segments, config, logger)
    if config.translation_provider == "openai":
        return _translate_openai(segments, config, logger)
    raise ValueError(f"Unknown translation provider: {config.translation_provider}")


def _translate_deep_translator(segments: list[dict], config: PipelineConfig, logger: logging.Logger) -> list[dict]:
    try:
        from deep_translator import GoogleTranslator
    except ImportError as exc:
        raise RuntimeError("Missing Python package deep-translator. Run: py -m pip install -r requirements.txt") from exc

    translator = GoogleTranslator(source="auto", target=config.target_language)
    out = []
    for seg in segments:
        translated = translator.translate(seg["text"])
        logger.info("Translated segment id=%s chars=%s", seg["id"], len(translated or ""))
        out.append({**seg, "translated_text": translated or seg["text"]})
    return out


def _translate_openai(segments: list[dict], config: PipelineConfig, logger: logging.Logger) -> list[dict]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing Python package openai. Run: py -m pip install -r requirements.txt") from exc

    client = OpenAI()
    out = []
    for seg in segments:
        duration = max(0.1, float(seg["end"]) - float(seg["start"]))
        prompt = (
            f"Translate this subtitle segment to {config.target_language}. "
            f"Keep it natural for voice dubbing and concise enough to speak in about {duration:.1f} seconds. "
            "Return only the translated sentence.\n\n"
            f"{seg['text']}"
        )
        response = client.chat.completions.create(
            model=config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        translated = response.choices[0].message.content.strip()
        logger.info("OpenAI translated segment id=%s duration=%.3f chars=%s", seg["id"], duration, len(translated))
        out.append({**seg, "translated_text": translated})
    return out


def _build_dubbed_audio(segments: list[dict], output_path: Path, config: PipelineConfig, logger: logging.Logger, job_dir: Path) -> None:
    if not segments:
        raise RuntimeError("No transcript segments found.")

    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Missing Python package edge-tts. Run: py -m pip install -r requirements.txt") from exc

    tts_dir = job_dir / "work" / "tts"
    fit_dir = job_dir / "work" / "fit"
    tts_dir.mkdir(parents=True, exist_ok=True)
    fit_dir.mkdir(parents=True, exist_ok=True)

    if config.timing_mode == "no_cut_sequential":
        _build_sequential_dubbed_audio(segments, output_path, config, logger, job_dir, edge_tts)
        return

    for seg in segments:
        text = seg.get("translated_text") or seg["text"]
        raw_mp3 = tts_dir / f"seg_{seg['id']:05d}.mp3"
        fit_wav = fit_dir / f"seg_{seg['id']:05d}.wav"
        logger.info("TTS segment id=%s start=%.3f end=%.3f text_chars=%s", seg["id"], seg["start"], seg["end"], len(text))
        _save_tts_with_retries(edge_tts, text, config.tts_voice, raw_mp3, logger)
        _fit_segment_audio(raw_mp3, fit_wav, max(0.1, seg["end"] - seg["start"]), config.max_tempo, logger)

    total_duration = max(float(seg["end"]) for seg in segments)
    silence_path = job_dir / "work" / "silence.wav"
    _run([
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=44100",
        "-t",
        f"{total_duration:.3f}",
        str(silence_path),
    ], logger)

    inputs = ["-i", str(silence_path)]
    filters = []
    mix_inputs = ["[0:a]"]
    for index, seg in enumerate(segments, start=1):
        fit_wav = fit_dir / f"seg_{seg['id']:05d}.wav"
        inputs.extend(["-i", str(fit_wav)])
        delay_ms = max(0, int(float(seg["start"]) * 1000))
        label = f"a{index}"
        filters.append(f"[{index}:a]adelay={delay_ms}:all=1[{label}]")
        mix_inputs.append(f"[{label}]")

    filter_complex = ";".join(filters + [f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0[out]"])
    _run([
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-ac",
        "2",
        "-ar",
        "44100",
        str(output_path),
    ], logger)
    logger.info("Dubbed audio created: %s", output_path)


def _build_sequential_dubbed_audio(segments: list[dict], output_path: Path, config: PipelineConfig, logger: logging.Logger, job_dir: Path, edge_tts_module) -> None:
    tts_dir = job_dir / "work" / "tts"
    fit_dir = job_dir / "work" / "fit"
    tts_dir.mkdir(parents=True, exist_ok=True)
    fit_dir.mkdir(parents=True, exist_ok=True)

    scheduled = []
    cursor = 0.0
    for seg in segments:
        text = seg.get("translated_text") or seg["text"]
        raw_mp3 = tts_dir / f"seg_{seg['id']:05d}.mp3"
        wav_path = fit_dir / f"seg_{seg['id']:05d}.wav"
        original_duration = max(0.1, float(seg["end"]) - float(seg["start"]))

        logger.info(
            "TTS sequential segment id=%s original_start=%.3f original_end=%.3f text_chars=%s",
            seg["id"],
            seg["start"],
            seg["end"],
            len(text),
        )
        _save_tts_with_retries(edge_tts_module, text, config.tts_voice, raw_mp3, logger)
        raw_duration = _duration(raw_mp3, logger)
        tempo = raw_duration / original_duration if original_duration > 0 else 1.0
        if tempo > 1.0:
            applied_tempo = min(tempo, config.max_tempo)
            audio_filter = f"atempo={applied_tempo:.5f}"
        else:
            audio_filter = "anull"
        _run([
            "ffmpeg",
            "-y",
            "-i",
            str(raw_mp3),
            "-filter:a",
            audio_filter,
            "-ac",
            "1",
            "-ar",
            "44100",
            str(wav_path),
        ], logger)

        fitted_duration = _duration(wav_path, logger)
        scheduled_start = max(float(seg["start"]), cursor)
        scheduled.append({**seg, "scheduled_start": scheduled_start, "audio_path": wav_path, "audio_duration": fitted_duration})
        cursor = scheduled_start + fitted_duration
        logger.info(
            "Sequential timing id=%s original_start=%.3f scheduled_start=%.3f fitted_duration=%.3f cursor=%.3f",
            seg["id"],
            seg["start"],
            scheduled_start,
            fitted_duration,
            cursor,
        )

    total_duration = max(cursor, max(float(seg["end"]) for seg in segments))
    schedule_path = job_dir / "output" / "timing_schedule.json"
    _write_json(schedule_path, [
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
    logger.info("Sequential schedule total duration: %.3f. Saved: %s", total_duration, schedule_path)

    silence_path = job_dir / "work" / "silence.wav"
    _run([
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=44100",
        "-t",
        f"{total_duration:.3f}",
        str(silence_path),
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
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-ac",
        "2",
        "-ar",
        "44100",
        str(output_path),
    ], logger)
    logger.info("Sequential dubbed audio created: %s", output_path)


async def _edge_tts_save(edge_tts_module, text: str, voice: str, output_path: Path) -> None:
    communicate = edge_tts_module.Communicate(text, voice)
    await communicate.save(str(output_path))


def _save_tts_with_retries(edge_tts_module, text: str, voice: str, output_path: Path, logger: logging.Logger) -> None:
    last_error = None
    clean_text = " ".join(text.split())
    for attempt in range(1, 4):
        try:
            if output_path.exists():
                output_path.unlink()
            logger.info("edge-tts attempt=%s voice=%s output=%s", attempt, voice, output_path)
            asyncio.run(_edge_tts_save(edge_tts_module, clean_text, voice, output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                return
            raise RuntimeError(f"edge-tts produced empty file: {output_path}")
        except Exception as exc:
            last_error = exc
            logger.warning("edge-tts failed attempt=%s error=%s", attempt, exc)
            time.sleep(2 * attempt)
    raise RuntimeError(f"edge-tts failed after retries: {last_error}") from last_error


def _fit_segment_audio(input_path: Path, output_path: Path, target_duration: float, max_tempo: float, logger: logging.Logger) -> None:
    source_duration = _duration(input_path, logger)
    tempo = source_duration / target_duration if target_duration > 0 else 1.0
    logger.info(
        "Fit segment audio source_duration=%.3f target_duration=%.3f raw_tempo=%.3f",
        source_duration,
        target_duration,
        tempo,
    )
    if tempo > 1.0:
        applied_tempo = min(tempo, max_tempo)
        audio_filter = f"atempo={applied_tempo:.5f},apad,atrim=0:{target_duration:.3f}"
    else:
        audio_filter = f"apad,atrim=0:{target_duration:.3f}"

    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-filter:a",
        audio_filter,
        "-ac",
        "1",
        "-ar",
        "44100",
        str(output_path),
    ], logger)


def _mux_video(
    input_video: Path,
    original_audio: Path,
    dubbed_audio: Path,
    final_video: Path,
    keep_original_audio: bool,
    original_audio_volume: float,
    shortest: bool,
    video_when_audio_longer: str,
    logger: logging.Logger,
) -> None:
    if keep_original_audio and original_audio_volume > 0:
        mixed_audio = final_video.parent / "mixed.wav"
        volume = max(0.0, float(original_audio_volume))
        _run([
            "ffmpeg",
            "-y",
            "-i",
            str(original_audio),
            "-i",
            str(dubbed_audio),
            "-filter_complex",
            f"[0:a]volume={volume:.5f}[a0];[a0][1:a]amix=inputs=2:normalize=0[out]",
            "-map",
            "[out]",
            "-ac",
            "2",
            "-ar",
            "44100",
            str(mixed_audio),
        ], logger)
        audio_for_mux = mixed_audio
    else:
        audio_for_mux = dubbed_audio

    video_for_mux = input_video
    video_duration = _duration(input_video, logger)
    audio_duration = _duration(audio_for_mux, logger)
    if not shortest and audio_duration > video_duration + 0.25 and video_when_audio_longer == "stretch_video":
        stretched_video = final_video.parent / "stretched_video.mp4"
        ratio = audio_duration / video_duration
        logger.info(
            "Stretching video to match audio: video_duration=%.3f audio_duration=%.3f ratio=%.5f",
            video_duration,
            audio_duration,
            ratio,
        )
        _run([
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-filter:v",
            f"setpts={ratio:.8f}*PTS",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(stretched_video),
        ], logger)
        video_for_mux = stretched_video

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_for_mux),
        "-i",
        str(audio_for_mux),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
    ]
    if shortest:
        cmd.append("-shortest")
    cmd.append(str(final_video))
    _run(cmd, logger)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _tail(path: Path, lines: int = 160) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _tail_job_logs(job_dir: Path) -> str:
    parts = []
    for label, path in [
        ("pipeline.log", job_dir / "logs" / "pipeline.log"),
        ("colab_asr.log", job_dir / "logs" / "colab_asr.log"),
        ("omnivoice_tts.log", job_dir / "logs" / "omnivoice_tts.log"),
    ]:
        if path.exists():
            parts.append(f"===== {label} =====\n{_tail(path, 100)}")
    return "\n\n".join(parts)


def _repair_transcript_text(segments: list[dict], logger: logging.Logger) -> list[dict]:
    repaired_count = 0
    repaired_segments = []
    for seg in segments:
        item = dict(seg)
        for key in ["text", "translated_text"]:
            value = item.get(key)
            if isinstance(value, str):
                repaired = _repair_mojibake(value)
                if repaired != value:
                    item[key] = repaired
                    repaired_count += 1
        repaired_segments.append(item)
    if repaired_count:
        logger.info("Repaired mojibake text fields: %s", repaired_count)
    return repaired_segments


def _repair_mojibake(text: str) -> str:
    if not any(marker in text for marker in ["Ã", "Â", "â", "ä", "å", "ç", "è", "é"]):
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    if _looks_better_text(repaired, text):
        return repaired
    return text


def _looks_better_text(candidate: str, original: str) -> bool:
    bad_markers = ["Ã", "Â", "â", "ä", "å", "ç", "è", "é", "�"]
    original_bad = sum(original.count(marker) for marker in bad_markers)
    candidate_bad = sum(candidate.count(marker) for marker in bad_markers)
    return candidate_bad < original_bad
