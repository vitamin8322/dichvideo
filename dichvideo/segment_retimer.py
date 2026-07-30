from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SegmentRetimeConfig:
    min_video_ratio: float = 0.70
    max_video_ratio: float = 1.40
    min_audio_tempo: float = 0.75
    max_audio_tempo: float = 1.80
    keep_gaps: bool = True
    original_audio_volume: float = 0.0
    crf: int = 20
    preset: str = "veryfast"
    output_fps: int = 30


@dataclass
class SegmentRetimeUpdate:
    status: str
    log_text: str
    video_path: str | None = None
    schedule_path: str | None = None
    log_path: str | None = None
    job_path: str | None = None


@dataclass
class VideoTimelinePiece:
    start: float
    end: float
    ratio: float
    duration: float


def retime_video_to_audio_segments(
    video_path: Path,
    srt_path: Path,
    audio_paths: list[Path],
    jobs_root: Path,
    config: SegmentRetimeConfig,
) -> Iterable[SegmentRetimeUpdate]:
    job_dir = _create_job_dir(jobs_root)
    log_path = job_dir / "logs" / "segment_retimer.log"
    logger = _setup_logger(log_path)

    def update(message: str, video: Path | None = None, schedule: Path | None = None) -> SegmentRetimeUpdate:
        logger.info(message)
        return SegmentRetimeUpdate(
            status=message,
            log_text=_tail(log_path),
            video_path=str(video) if video and video.exists() else None,
            schedule_path=str(schedule) if schedule and schedule.exists() else None,
            log_path=str(log_path),
            job_path=str(job_dir),
        )

    try:
        _check_binary("ffmpeg")
        _check_binary("ffprobe")
        if not video_path.exists():
            raise RuntimeError(f"Video not found: {video_path}")
        if not srt_path.exists():
            raise RuntimeError(f"SRT not found: {srt_path}")
        if not audio_paths:
            raise RuntimeError("No audio segment files provided.")

        input_video = job_dir / "input" / video_path.name
        input_srt = job_dir / "input" / srt_path.name
        shutil.copy2(video_path, input_video)
        shutil.copy2(srt_path, input_srt)
        copied_audio_paths = []
        for index, audio_path in enumerate(_sort_audio_paths(audio_paths), start=1):
            copied = job_dir / "input" / "audio_segments" / f"{index:04d}{audio_path.suffix.lower() or '.wav'}"
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio_path, copied)
            copied_audio_paths.append(copied)

        yield update(f"Created segment retime job: {job_dir}")

        segments = parse_srt(input_srt.read_text(encoding="utf-8-sig"))
        if len(copied_audio_paths) < len(segments):
            raise RuntimeError(f"Need at least {len(segments)} audio segments, got {len(copied_audio_paths)}.")
        if len(copied_audio_paths) > len(segments):
            logger.warning("More audio files than SRT segments. Extra files will be ignored: %s > %s", len(copied_audio_paths), len(segments))
            copied_audio_paths = copied_audio_paths[:len(segments)]

        video_duration = _duration(input_video, logger)
        _validate_segments(segments, video_duration)
        has_original_audio = config.original_audio_volume > 0 and _has_audio(input_video, logger)
        if config.original_audio_volume > 0 and not has_original_audio:
            logger.warning("Original audio volume was requested, but source video has no readable audio stream.")
        yield update(f"Loaded {len(segments)} SRT segment(s). Video duration: {video_duration:.3f}s")

        video_timeline = []
        audio_piece_list = []
        original_audio_piece_list = []
        schedule = []
        cursor = 0.0
        previous_end = 0.0

        for index, (segment, audio_path) in enumerate(zip(segments, copied_audio_paths), start=1):
            if config.keep_gaps and segment["start"] - previous_end >= 0.05:
                gap_duration = segment["start"] - previous_end
                gap_audio = job_dir / "work" / "audio_pieces" / f"{len(audio_piece_list):05d}_gap.wav"
                video_timeline.append(VideoTimelinePiece(previous_end, segment["start"], 1.0, gap_duration))
                _make_silence(gap_audio, gap_duration, logger)
                audio_piece_list.append(gap_audio)
                if has_original_audio:
                    original_gap_audio = job_dir / "work" / "original_audio_pieces" / f"{len(original_audio_piece_list):05d}_gap.wav"
                    _extract_original_audio_piece(input_video, previous_end, segment["start"], 1.0, gap_duration, original_gap_audio, logger)
                    original_audio_piece_list.append(original_gap_audio)
                schedule.append({
                    "type": "gap",
                    "original_start": round(previous_end, 3),
                    "original_end": round(segment["start"], 3),
                    "output_start": round(cursor, 3),
                    "output_end": round(cursor + gap_duration, 3),
                    "duration": round(gap_duration, 3),
                })
                cursor += gap_duration

            original_video_duration = max(0.001, segment["end"] - segment["start"])
            original_audio_duration = _duration(audio_path, logger)
            desired_video_ratio = original_audio_duration / original_video_duration
            video_ratio = _clamp(desired_video_ratio, config.min_video_ratio, config.max_video_ratio)
            target_duration = original_video_duration * video_ratio

            if abs(target_duration - original_audio_duration) <= 0.03:
                audio_tempo = 1.0
            else:
                audio_tempo = _clamp(original_audio_duration / target_duration, config.min_audio_tempo, config.max_audio_tempo)

            audio_piece = job_dir / "work" / "audio_pieces" / f"{len(audio_piece_list):05d}_seg_{index:04d}.wav"
            video_timeline.append(VideoTimelinePiece(segment["start"], segment["end"], video_ratio, target_duration))
            _retime_audio_piece(audio_path, audio_tempo, target_duration, audio_piece, logger)
            actual_audio_piece_duration = _duration(audio_piece, logger)
            if abs(target_duration - actual_audio_piece_duration) > 0.08:
                logger.warning(
                    "Rendered segment duration mismatch index=%s target=%.3fs video_piece=%.3fs audio_piece=%.3fs",
                    index,
                    target_duration,
                    target_duration,
                    actual_audio_piece_duration,
                )
            if has_original_audio:
                original_audio_piece = job_dir / "work" / "original_audio_pieces" / f"{len(original_audio_piece_list):05d}_seg_{index:04d}.wav"
                _extract_original_audio_piece(
                    input_video,
                    segment["start"],
                    segment["end"],
                    1.0 / video_ratio,
                    target_duration,
                    original_audio_piece,
                    logger,
                )
                original_audio_piece_list.append(original_audio_piece)

            audio_piece_list.append(audio_piece)
            schedule.append({
                "type": "segment",
                "index": segment["index"],
                "text": segment["text"],
                "original_start": round(segment["start"], 3),
                "original_end": round(segment["end"], 3),
                "original_video_duration": round(original_video_duration, 3),
                "original_audio_duration": round(original_audio_duration, 3),
                "desired_video_ratio": round(desired_video_ratio, 5),
                "video_ratio": round(video_ratio, 5),
                "target_duration": round(target_duration, 3),
                "audio_tempo": round(audio_tempo, 5),
                "actual_video_piece_duration": round(target_duration, 3),
                "actual_audio_piece_duration": round(actual_audio_piece_duration, 3),
                "output_start": round(cursor, 3),
                "output_end": round(cursor + target_duration, 3),
                "audio_file": str(audio_path.name),
            })
            logger.info(
                "Segment %s video=%.3fs audio=%.3fs desired_ratio=%.5f video_ratio=%.5f target=%.3fs audio_tempo=%.5f",
                index,
                original_video_duration,
                original_audio_duration,
                desired_video_ratio,
                video_ratio,
                target_duration,
                audio_tempo,
            )
            cursor += target_duration
            previous_end = segment["end"]
            yield update(f"Processed segment {index}/{len(segments)}")

        if config.keep_gaps and video_duration - previous_end >= 0.05:
            gap_duration = video_duration - previous_end
            gap_audio = job_dir / "work" / "audio_pieces" / f"{len(audio_piece_list):05d}_tail.wav"
            video_timeline.append(VideoTimelinePiece(previous_end, video_duration, 1.0, gap_duration))
            _make_silence(gap_audio, gap_duration, logger)
            audio_piece_list.append(gap_audio)
            if has_original_audio:
                original_tail_audio = job_dir / "work" / "original_audio_pieces" / f"{len(original_audio_piece_list):05d}_tail.wav"
                _extract_original_audio_piece(input_video, previous_end, video_duration, 1.0, gap_duration, original_tail_audio, logger)
                original_audio_piece_list.append(original_tail_audio)
            schedule.append({
                "type": "tail",
                "original_start": round(previous_end, 3),
                "original_end": round(video_duration, 3),
                "output_start": round(cursor, 3),
                "output_end": round(cursor + gap_duration, 3),
                "duration": round(gap_duration, 3),
            })
            cursor += gap_duration

        schedule_path = job_dir / "output" / "retime_schedule.json"
        _write_json(schedule_path, {
            "source_video": str(input_video),
            "source_srt": str(input_srt),
            "config": config.__dict__,
            "output_duration": round(cursor, 3),
            "items": schedule,
        })

        yield update("Rendering retimed video timeline and concatenating audio...")
        retimed_video = job_dir / "output" / "retimed_video.mp4"
        retimed_audio = job_dir / "output" / "retimed_audio.wav"
        _render_video_timeline(input_video, video_timeline, retimed_video, config, logger, job_dir)
        _concat_media(audio_piece_list, retimed_audio, "audio", logger, job_dir)
        audio_for_mux = retimed_audio
        if has_original_audio:
            original_retimed_audio = job_dir / "output" / "original_retimed_audio.wav"
            mixed_audio = job_dir / "output" / "mixed_audio.wav"
            _concat_media(original_audio_piece_list, original_retimed_audio, "original_audio", logger, job_dir)
            _mix_audio(original_retimed_audio, retimed_audio, mixed_audio, config.original_audio_volume, logger)
            audio_for_mux = mixed_audio

        final_video = job_dir / "output" / "final.mp4"
        _run([
            "ffmpeg", "-y",
            "-i", str(retimed_video),
            "-i", str(audio_for_mux),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(final_video),
        ], logger)
        yield update("Done", video=final_video, schedule=schedule_path)
    except Exception:
        logger.exception("Segment retime failed")
        yield SegmentRetimeUpdate(
            status="Segment retime failed. Xem log de biet chi tiet.",
            log_text=_tail(log_path),
            log_path=str(log_path),
            job_path=str(job_dir),
        )


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


def _extract_video_piece(input_video: Path, start: float, end: float, output_path: Path, config: SegmentRetimeConfig, logger: logging.Logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.001, end - start)
    video_filter = (
        f"trim=start={start:.6f}:end={end:.6f},"
        "setpts=PTS-STARTPTS,"
        f"fps={config.output_fps},"
        "tpad=stop_mode=clone:stop_duration=1,"
        f"trim=duration={duration:.6f},"
        "setpts=PTS-STARTPTS"
    )
    _run([
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-an",
        "-filter:v", video_filter,
        "-c:v", "libx264",
        "-preset", config.preset,
        "-crf", str(config.crf),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ], logger)


def _extract_and_retime_video_piece(input_video: Path, start: float, end: float, ratio: float, target_duration: float, output_path: Path, config: SegmentRetimeConfig, logger: logging.Logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_duration = max(0.001, target_duration)
    video_filter = (
        f"trim=start={start:.6f}:end={end:.6f},"
        "setpts=PTS-STARTPTS,"
        f"setpts={ratio:.8f}*PTS,"
        f"fps={config.output_fps},"
        "tpad=stop_mode=clone:stop_duration=1,"
        f"trim=duration={target_duration:.6f},"
        "setpts=PTS-STARTPTS"
    )
    _run([
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-an",
        "-filter:v", video_filter,
        "-c:v", "libx264",
        "-preset", config.preset,
        "-crf", str(config.crf),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ], logger)


def _render_video_timeline(
    input_video: Path,
    pieces: list[VideoTimelinePiece],
    output_path: Path,
    config: SegmentRetimeConfig,
    logger: logging.Logger,
    job_dir: Path,
) -> None:
    if not pieces:
        raise RuntimeError("No video timeline pieces to render.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_script_path = job_dir / "work" / "video_timeline_filter.txt"
    filter_script_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    labels = []
    for index, piece in enumerate(pieces):
        duration = max(0.001, piece.duration)
        label = f"v{index}"
        labels.append(f"[{label}]")
        lines.append(
            f"[0:v]trim=start={piece.start:.6f}:end={piece.end:.6f},"
            "setpts=PTS-STARTPTS,"
            f"setpts={piece.ratio:.8f}*PTS,"
            f"fps={config.output_fps},"
            "tpad=stop_mode=clone:stop_duration=1,"
            f"trim=duration={duration:.6f},"
            f"setpts=PTS-STARTPTS[{label}]"
        )

    lines.append("".join(labels) + f"concat=n={len(pieces)}:v=1:a=0[vout]")
    filter_script_path.write_text(";\n".join(lines), encoding="utf-8")

    _run([
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-filter_complex_script", str(filter_script_path),
        "-map", "[vout]",
        "-an",
        "-c:v", "libx264",
        "-preset", config.preset,
        "-crf", str(config.crf),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ], logger)


def _retime_audio_piece(input_audio: Path, tempo: float, target_duration: float, output_path: Path, logger: logging.Logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if abs(tempo - 1.0) > 0.001:
        filters.append(_atempo_filter(tempo))
    filters.extend(["apad", f"atrim=0:{target_duration:.3f}"])
    _run([
        "ffmpeg", "-y",
        "-i", str(input_audio),
        "-filter:a", ",".join(filters),
        "-ac", "2",
        "-ar", "44100",
        str(output_path),
    ], logger)


def _extract_original_audio_piece(input_video: Path, start: float, end: float, tempo: float, target_duration: float, output_path: Path, logger: logging.Logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_duration = max(0.001, target_duration)
    filters = [f"atrim=start={start:.6f}:end={end:.6f}", "asetpts=PTS-STARTPTS"]
    if abs(tempo - 1.0) > 0.001:
        filters.append(_atempo_filter(tempo))
    filters.extend(["apad", f"atrim=0:{target_duration:.6f}", "asetpts=PTS-STARTPTS"])
    _run([
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vn",
        "-filter:a", ",".join(filters),
        "-ac", "2",
        "-ar", "44100",
        str(output_path),
    ], logger)


def _make_silence(output_path: Path, duration: float, logger: logging.Logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", f"{duration:.3f}",
        str(output_path),
    ], logger)


def _concat_media(paths: list[Path], output_path: Path, kind: str, logger: logging.Logger, job_dir: Path) -> None:
    list_path = job_dir / "work" / f"{kind}_concat.txt"
    list_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("".join(f"file '{path.resolve().as_posix()}'\n" for path in paths), encoding="utf-8")
    if kind == "video":
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(output_path)]
    else:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c:a", "pcm_s16le", "-ac", "2", "-ar", "44100", str(output_path)]
    _run(cmd, logger)


def _mix_audio(original_audio: Path, dubbed_audio: Path, output_path: Path, original_volume: float, logger: logging.Logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    volume = max(0.0, float(original_volume))
    _run([
        "ffmpeg", "-y",
        "-i", str(original_audio),
        "-i", str(dubbed_audio),
        "-filter_complex", f"[0:a]volume={volume:.5f}[a0];[a0][1:a]amix=inputs=2:normalize=0[out]",
        "-map", "[out]",
        "-ac", "2",
        "-ar", "44100",
        str(output_path),
    ], logger)


def _atempo_filter(tempo: float) -> str:
    parts = []
    remaining = tempo
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.5f}")
    return ",".join(parts)


def _sort_audio_paths(paths: list[Path]) -> list[Path]:
    def key(path: Path):
        numbers = re.findall(r"\d+", path.stem)
        return (int(numbers[-1]) if numbers else 10**9, path.name.lower())
    return sorted(paths, key=key)


def _validate_segments(segments: list[dict], video_duration: float) -> None:
    previous_end = 0.0
    for index, segment in enumerate(segments, start=1):
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start:
            raise RuntimeError(
                f"SRT segment {index} has invalid timing: start={_format_seconds(start)}, end={_format_seconds(end)}."
            )
        if start < previous_end - 0.001:
            raise RuntimeError(
                f"SRT segment {index} starts before the previous segment ends: "
                f"previous_end={_format_seconds(previous_end)}, start={_format_seconds(start)}."
            )
        if end > video_duration + 0.25:
            raise RuntimeError(
                f"SRT segment {index} ends after the video duration: "
                f"end={_format_seconds(end)}, video_duration={_format_seconds(video_duration)}."
            )
        previous_end = max(previous_end, end)


def _format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def _duration(path: Path, logger: logging.Logger) -> float:
    completed = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ], logger)
    return float(completed.stdout.strip())


def _has_audio(path: Path, logger: logging.Logger) -> bool:
    completed = _run([
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path),
    ], logger)
    return bool(completed.stdout.strip())


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _create_job_dir(jobs_root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jobs_root.mkdir(parents=True, exist_ok=True)
    for counter in range(1000):
        suffix = "" if counter == 0 else f"_{counter:03d}"
        job_dir = jobs_root / f"segment_retime_{stamp}{suffix}"
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
            for child in ["input", "work", "output", "logs"]:
                (job_dir / child).mkdir(parents=True, exist_ok=True)
            return job_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create unique segment retime job folder under: {jobs_root}")


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"dichvideo.segment_retimer.{log_path.parent.parent.name}")
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
        raise RuntimeError(f"Missing dependency: {name}. Hay cai ffmpeg va them vao PATH.")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _tail(path: Path, lines: int = 160) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
