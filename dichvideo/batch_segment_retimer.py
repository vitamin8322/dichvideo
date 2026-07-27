from __future__ import annotations

import json
import logging
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .segment_retimer import SegmentRetimeConfig, retime_video_to_audio_segments


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
AUDIO_FOLDER_NAMES = ["segments", "audio_segments", "mix_segments", "audios"]


@dataclass
class BatchSegmentRetimeConfig:
    segment_config: SegmentRetimeConfig
    max_parallel_jobs: int = 2


@dataclass
class BatchSegmentRetimeUpdate:
    status: str
    log_text: str
    summary_path: str | None = None
    zip_path: str | None = None
    batch_job_path: str | None = None


@dataclass
class BatchTask:
    name: str
    folder: Path
    video_path: Path
    srt_path: Path
    audio_paths: list[Path]


def run_batch_segment_retime(
    batch_root: Path,
    jobs_root: Path,
    config: BatchSegmentRetimeConfig,
) -> Iterable[BatchSegmentRetimeUpdate]:
    batch_dir = _create_batch_dir(jobs_root)
    log_path = batch_dir / "logs" / "batch_segment_retimer.log"
    logger = _setup_logger(log_path)
    summary_path = batch_dir / "output" / "batch_summary.json"
    zip_path = batch_dir / "output" / "batch_final_videos.zip"

    def update(message: str) -> BatchSegmentRetimeUpdate:
        logger.info(message)
        return BatchSegmentRetimeUpdate(
            status=message,
            log_text=_tail(log_path),
            summary_path=str(summary_path) if summary_path.exists() else None,
            zip_path=str(zip_path) if zip_path.exists() else None,
            batch_job_path=str(batch_dir),
        )

    try:
        if not batch_root.exists():
            raise RuntimeError(f"Batch root not found: {batch_root}")
        if not batch_root.is_dir():
            raise RuntimeError(f"Batch root must be a folder: {batch_root}")

        tasks, invalid_results = _discover_tasks(batch_root)
        results = invalid_results[:]
        _write_summary(summary_path, batch_root, batch_dir, config, results)
        yield update(f"Found {len(tasks)} valid job(s), {len(invalid_results)} invalid folder(s).")

        if not tasks:
            yield update("Batch failed: khong tim thay folder hop le de xu ly.")
            return

        max_workers = max(1, min(4, int(config.max_parallel_jobs)))
        logger.info("Starting batch with max_parallel_jobs=%s", max_workers)
        child_jobs_root = batch_dir / "jobs"
        final_videos_dir = batch_dir / "output" / "final_videos"
        schedules_dir = batch_dir / "output" / "schedules"
        child_logs_dir = batch_dir / "output" / "logs"
        for path in [final_videos_dir, schedules_dir, child_logs_dir]:
            path.mkdir(parents=True, exist_ok=True)

        yield update(f"Starting {len(tasks)} job(s) with {max_workers} parallel worker(s)...")
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(_run_one_task, task, child_jobs_root, config.segment_config, logger): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result["status"] == "done":
                        _collect_result_artifacts(result, task.name, final_videos_dir, schedules_dir, child_logs_dir, logger)
                    results.append(result)
                except Exception as exc:
                    logger.exception("Batch task crashed: %s", task.name)
                    results.append({
                        "name": task.name,
                        "input_folder": str(task.folder),
                        "status": "failed",
                        "error": str(exc),
                    })
                completed_count += 1
                _write_summary(summary_path, batch_root, batch_dir, config, results)
                done = sum(1 for item in results if item.get("status") == "done")
                failed = sum(1 for item in results if item.get("status") in {"failed", "invalid"})
                yield update(f"Completed {completed_count}/{len(tasks)}. Done={done}, failed_or_invalid={failed}.")

        if any(final_videos_dir.glob("*.mp4")):
            archive_base = str(zip_path.with_suffix(""))
            created_zip = shutil.make_archive(archive_base, "zip", final_videos_dir)
            logger.info("Created final videos zip: %s", created_zip)

        _write_summary(summary_path, batch_root, batch_dir, config, results)
        done = sum(1 for item in results if item.get("status") == "done")
        failed = sum(1 for item in results if item.get("status") in {"failed", "invalid"})
        yield update(f"Batch done. Done={done}, failed_or_invalid={failed}.")
    except Exception:
        logger.exception("Batch segment retime failed")
        yield BatchSegmentRetimeUpdate(
            status="Batch segment retime failed. Xem log de biet chi tiet.",
            log_text=_tail(log_path),
            summary_path=str(summary_path) if summary_path.exists() else None,
            zip_path=str(zip_path) if zip_path.exists() else None,
            batch_job_path=str(batch_dir),
        )


def run_batch_segment_retime_uploads(
    video_paths: list[Path],
    srt_paths: list[Path],
    audio_archive_paths: list[Path],
    jobs_root: Path,
    config: BatchSegmentRetimeConfig,
) -> Iterable[BatchSegmentRetimeUpdate]:
    if not video_paths:
        raise RuntimeError("No video files uploaded.")
    if len(video_paths) != len(srt_paths):
        raise RuntimeError(f"Video count and SRT count must match: {len(video_paths)} != {len(srt_paths)}.")
    if len(video_paths) != len(audio_archive_paths):
        raise RuntimeError(f"Video count and audio ZIP count must match: {len(video_paths)} != {len(audio_archive_paths)}.")

    staging_root = _create_upload_staging_dir(jobs_root)
    for index, (video_path, srt_path, audio_archive_path) in enumerate(zip(video_paths, srt_paths, audio_archive_paths), start=1):
        video_path = Path(video_path).resolve()
        srt_path = Path(srt_path).resolve()
        audio_archive_path = Path(audio_archive_path).resolve()
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError(f"Unsupported video file: {video_path.name}")
        if srt_path.suffix.lower() != ".srt":
            raise RuntimeError(f"Unsupported SRT file: {srt_path.name}")
        if audio_archive_path.suffix.lower() != ".zip":
            raise RuntimeError(f"Audio segments must be uploaded as .zip: {audio_archive_path.name}")

        task_name = _safe_name(video_path.stem) or f"video_{index:03d}"
        task_folder = staging_root / f"{index:03d}_{task_name}"
        segments_dir = task_folder / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, task_folder / f"input{video_path.suffix.lower()}")
        shutil.copy2(srt_path, task_folder / "input.srt")
        extracted_count = _extract_audio_archive(audio_archive_path, segments_dir)
        if extracted_count == 0:
            raise RuntimeError(f"No audio segment files found in ZIP: {audio_archive_path.name}")

    yield from run_batch_segment_retime(staging_root, jobs_root, config)


def _run_one_task(task: BatchTask, jobs_root: Path, config: SegmentRetimeConfig, batch_logger: logging.Logger) -> dict:
    batch_logger.info(
        "Task start name=%s video=%s srt=%s audio_count=%s",
        task.name,
        task.video_path,
        task.srt_path,
        len(task.audio_paths),
    )
    last_update = None
    for update in retime_video_to_audio_segments(
        task.video_path,
        task.srt_path,
        task.audio_paths,
        jobs_root,
        config,
    ):
        last_update = update

    if last_update and last_update.video_path:
        return {
            "name": task.name,
            "input_folder": str(task.folder),
            "video": str(task.video_path),
            "srt": str(task.srt_path),
            "audio_count": len(task.audio_paths),
            "status": "done",
            "final_video": last_update.video_path,
            "schedule_path": last_update.schedule_path,
            "log_path": last_update.log_path,
            "child_job_path": last_update.job_path,
        }

    return {
        "name": task.name,
        "input_folder": str(task.folder),
        "video": str(task.video_path),
        "srt": str(task.srt_path),
        "audio_count": len(task.audio_paths),
        "status": "failed",
        "error": last_update.status if last_update else "No update returned.",
        "log_path": last_update.log_path if last_update else None,
        "child_job_path": last_update.job_path if last_update else None,
    }


def _discover_tasks(batch_root: Path) -> tuple[list[BatchTask], list[dict]]:
    tasks = []
    invalid = []
    for folder in sorted((path for path in batch_root.iterdir() if path.is_dir()), key=lambda item: item.name.lower()):
        try:
            tasks.append(_discover_one_task(folder))
        except Exception as exc:
            invalid.append({
                "name": folder.name,
                "input_folder": str(folder),
                "status": "invalid",
                "error": str(exc),
            })
    return tasks, invalid


def _discover_one_task(folder: Path) -> BatchTask:
    videos = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    srts = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".srt"]
    if len(videos) != 1:
        raise RuntimeError(f"Need exactly 1 video file in {folder.name}, found {len(videos)}.")
    if len(srts) != 1:
        raise RuntimeError(f"Need exactly 1 SRT file in {folder.name}, found {len(srts)}.")

    audio_folder = None
    for name in AUDIO_FOLDER_NAMES:
        candidate = folder / name
        if candidate.is_dir():
            audio_folder = candidate
            break
    if audio_folder is None:
        raise RuntimeError(f"Need an audio folder named one of: {', '.join(AUDIO_FOLDER_NAMES)}.")

    audio_paths = [
        path for path in audio_folder.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not audio_paths:
        raise RuntimeError(f"No audio segment files found in {audio_folder}.")

    return BatchTask(
        name=_safe_name(folder.name),
        folder=folder,
        video_path=videos[0],
        srt_path=srts[0],
        audio_paths=_sort_audio_paths(audio_paths),
    )


def _collect_result_artifacts(
    result: dict,
    task_name: str,
    final_videos_dir: Path,
    schedules_dir: Path,
    logs_dir: Path,
    logger: logging.Logger,
) -> None:
    final_video = Path(result["final_video"])
    copied_video = final_videos_dir / f"{task_name}.mp4"
    shutil.copy2(final_video, copied_video)
    result["collected_final_video"] = str(copied_video)

    if result.get("schedule_path"):
        copied_schedule = schedules_dir / f"{task_name}_retime_schedule.json"
        shutil.copy2(Path(result["schedule_path"]), copied_schedule)
        result["collected_schedule_path"] = str(copied_schedule)

    if result.get("log_path"):
        copied_log = logs_dir / f"{task_name}_segment_retimer.log"
        shutil.copy2(Path(result["log_path"]), copied_log)
        result["collected_log_path"] = str(copied_log)

    logger.info("Collected artifacts for task=%s final_video=%s", task_name, copied_video)


def _create_batch_dir(jobs_root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jobs_root.mkdir(parents=True, exist_ok=True)
    for counter in range(1000):
        suffix = "" if counter == 0 else f"_{counter:03d}"
        batch_dir = jobs_root / f"batch_segment_retime_{stamp}{suffix}"
        try:
            batch_dir.mkdir(parents=True, exist_ok=False)
            for child in ["output", "logs", "jobs"]:
                (batch_dir / child).mkdir(parents=True, exist_ok=True)
            return batch_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create unique batch folder under: {jobs_root}")


def _create_upload_staging_dir(jobs_root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jobs_root = jobs_root.resolve()
    jobs_root.mkdir(parents=True, exist_ok=True)
    for counter in range(1000):
        suffix = "" if counter == 0 else f"_{counter:03d}"
        staging_dir = jobs_root / f"batch_upload_staging_{stamp}{suffix}"
        try:
            staging_dir.mkdir(parents=True, exist_ok=False)
            return staging_dir.resolve()
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create unique batch upload staging folder under: {jobs_root}")


def _extract_audio_archive(archive_path: Path, output_dir: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            source_name = Path(item.filename)
            if "__MACOSX" in source_name.parts:
                continue
            if source_name.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            safe_name = _safe_name(source_name.stem) + source_name.suffix.lower()
            if not safe_name:
                safe_name = f"{count + 1:04d}.wav"
            output_path = output_dir / safe_name
            if output_path.exists():
                output_path = output_dir / f"{count + 1:04d}_{safe_name}"
            with archive.open(item) as source, output_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            count += 1
    return count


def _write_summary(
    summary_path: Path,
    batch_root: Path,
    batch_dir: Path,
    config: BatchSegmentRetimeConfig,
    results: list[dict],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "batch_root": str(batch_root),
        "batch_job_path": str(batch_dir),
        "config": {
            "max_parallel_jobs": config.max_parallel_jobs,
            "segment_config": config.segment_config.__dict__,
        },
        "results": results,
    }
    summary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"dichvideo.batch_segment_retimer.{log_path.parent.parent.name}")
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


def _sort_audio_paths(paths: list[Path]) -> list[Path]:
    def key(path: Path):
        numbers = re.findall(r"\d+", path.stem)
        return (int(numbers[-1]) if numbers else 10**9, path.name.lower())
    return sorted(paths, key=key)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return safe.strip("._-") or "video"


def _tail(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
