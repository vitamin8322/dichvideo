from pathlib import Path

import gradio as gr

from dichvideo.batch_segment_retimer import BatchSegmentRetimeConfig, run_batch_segment_retime_uploads
from dichvideo.pipeline import (
    PipelineConfig,
    finish_colab_asr_job,
    finish_remote_tts_job,
    package_omnivoice_input_job,
    prepare_colab_asr_job,
    read_job_status,
    run_pipeline,
    translate_colab_asr_job,
)
from dichvideo.segment_retimer import SegmentRetimeConfig, retime_video_to_audio_segments


ROOT = Path(__file__).resolve().parent
MAX_BATCH_SYNC_JOBS = 8


def _as_path(value) -> Path:
    if isinstance(value, (str, Path)):
        return Path(value)
    if hasattr(value, "name"):
        return Path(value.name)
    return Path(value)


def process_video(
    video_path,
    source_language,
    target_language,
    whisper_model,
    asr_device,
    compute_type,
    translation_provider,
    openai_model,
    tts_voice,
    chunk_minutes,
    max_tempo,
    keep_original_audio,
    original_audio_volume,
    timing_mode,
    video_when_audio_longer,
):
    if not video_path:
        raise gr.Error("Hay chon video dau vao.")

    config = PipelineConfig(
        source_language=None if source_language == "auto" else source_language,
        target_language=target_language.strip() or "vi",
        whisper_model=whisper_model,
        asr_device=asr_device,
        compute_type=compute_type,
        translation_provider=translation_provider,
        openai_model=openai_model.strip() or "gpt-4o-mini",
        tts_voice=tts_voice.strip() or "vi-VN-HoaiMyNeural",
        chunk_minutes=float(chunk_minutes),
        max_tempo=float(max_tempo),
        keep_original_audio=bool(keep_original_audio),
        original_audio_volume=float(original_audio_volume),
        timing_mode=timing_mode,
        video_when_audio_longer=video_when_audio_longer,
    )

    for update in run_pipeline(Path(video_path), ROOT / "jobs", config):
        yield update.video_path, update.status, update.log_text, update.transcript_path, update.log_path


def build_config(
    source_language,
    target_language,
    whisper_model,
    asr_device,
    compute_type,
    translation_provider,
    openai_model,
    tts_voice,
    chunk_minutes,
    max_tempo,
    keep_original_audio,
    original_audio_volume,
    timing_mode,
    video_when_audio_longer,
) -> PipelineConfig:
    return PipelineConfig(
        source_language=None if source_language == "auto" else source_language,
        target_language=target_language.strip() or "vi",
        whisper_model=whisper_model,
        asr_device=asr_device,
        compute_type=compute_type,
        translation_provider=translation_provider,
        openai_model=openai_model.strip() or "gpt-4o-mini",
        tts_voice=tts_voice.strip() or "vi-VN-HoaiMyNeural",
        chunk_minutes=float(chunk_minutes),
        max_tempo=float(max_tempo),
        keep_original_audio=bool(keep_original_audio),
        original_audio_volume=float(original_audio_volume),
        timing_mode=timing_mode,
        video_when_audio_longer=video_when_audio_longer,
    )


def prepare_colab_job(
    video_path,
    drive_jobs_root,
    source_language,
    target_language,
    whisper_model,
    asr_device,
    compute_type,
    translation_provider,
    openai_model,
    tts_voice,
    chunk_minutes,
    max_tempo,
    keep_original_audio,
    original_audio_volume,
    timing_mode,
    video_when_audio_longer,
):
    if not video_path:
        raise gr.Error("Hay chon video dau vao.")
    if not drive_jobs_root:
        raise gr.Error("Hay nhap folder jobs tren Google Drive local.")

    config = build_config(
        source_language,
        target_language,
        whisper_model,
        asr_device,
        compute_type,
        translation_provider,
        openai_model,
        tts_voice,
        chunk_minutes,
        max_tempo,
        keep_original_audio,
        original_audio_volume,
        timing_mode,
        video_when_audio_longer,
    )

    for update in prepare_colab_asr_job(Path(video_path), Path(drive_jobs_root), config):
        yield update.job_path, update.status, update.log_text, update.log_path


def check_job(job_path):
    if not job_path:
        raise gr.Error("Hay nhap job folder.")
    update = read_job_status(Path(job_path))
    return update.video_path, update.status, update.log_text, update.transcript_path, update.log_path


def finish_colab_job(
    job_path,
    source_language,
    target_language,
    whisper_model,
    asr_device,
    compute_type,
    translation_provider,
    openai_model,
    tts_voice,
    chunk_minutes,
    max_tempo,
    keep_original_audio,
    original_audio_volume,
    timing_mode,
    video_when_audio_longer,
):
    if not job_path:
        raise gr.Error("Hay nhap job folder da duoc Colab xu ly.")

    config = build_config(
        source_language,
        target_language,
        whisper_model,
        asr_device,
        compute_type,
        translation_provider,
        openai_model,
        tts_voice,
        chunk_minutes,
        max_tempo,
        keep_original_audio,
        original_audio_volume,
        timing_mode,
        video_when_audio_longer,
    )

    for update in finish_colab_asr_job(Path(job_path), config):
        yield update.video_path, update.status, update.log_text, update.transcript_path, update.log_path


def translate_asr_job(
    job_path,
    source_language,
    target_language,
    whisper_model,
    asr_device,
    compute_type,
    translation_provider,
    openai_model,
    tts_voice,
    chunk_minutes,
    max_tempo,
    keep_original_audio,
    original_audio_volume,
    timing_mode,
    video_when_audio_longer,
):
    if not job_path:
        raise gr.Error("Hay nhap job folder da co output/transcript.json.")

    config = build_config(
        source_language,
        target_language,
        whisper_model,
        asr_device,
        compute_type,
        translation_provider,
        openai_model,
        tts_voice,
        chunk_minutes,
        max_tempo,
        keep_original_audio,
        original_audio_volume,
        timing_mode,
        video_when_audio_longer,
    )

    for update in translate_colab_asr_job(Path(job_path), config):
        yield update.status, update.log_text, update.transcript_path, update.log_path


def package_omnivoice_job(job_path):
    if not job_path:
        raise gr.Error("Hay nhap job folder da co output/translated.json.")
    update = package_omnivoice_input_job(Path(job_path))
    return update.status, update.log_text, update.transcript_path, update.log_path


def finish_omnivoice_job(
    job_path,
    source_language,
    target_language,
    whisper_model,
    asr_device,
    compute_type,
    translation_provider,
    openai_model,
    tts_voice,
    chunk_minutes,
    max_tempo,
    keep_original_audio,
    original_audio_volume,
    timing_mode,
    video_when_audio_longer,
):
    if not job_path:
        raise gr.Error("Hay nhap job folder da co output/dubbed.wav tu OmniVoice Colab.")

    config = build_config(
        source_language,
        target_language,
        whisper_model,
        asr_device,
        compute_type,
        translation_provider,
        openai_model,
        tts_voice,
        chunk_minutes,
        max_tempo,
        keep_original_audio,
        original_audio_volume,
        timing_mode,
        video_when_audio_longer,
    )

    for update in finish_remote_tts_job(Path(job_path), config):
        yield update.video_path, update.status, update.log_text, update.transcript_path, update.log_path


def retime_srt_audio_segments(
    video_path,
    srt_file,
    audio_files,
    min_video_ratio,
    max_video_ratio,
    min_audio_tempo,
    max_audio_tempo,
    keep_gaps,
    original_audio_volume,
    crf,
    preset,
):
    if not video_path:
        raise gr.Error("Hay chon video goc.")
    if not srt_file:
        raise gr.Error("Hay upload file SRT goc.")
    if not audio_files:
        raise gr.Error("Hay upload cac audio segment.")
    if float(min_video_ratio) > float(max_video_ratio):
        raise gr.Error("Min video ratio phai nho hon hoac bang max video ratio.")
    if float(min_audio_tempo) > float(max_audio_tempo):
        raise gr.Error("Min audio tempo phai nho hon hoac bang max audio tempo.")

    config = SegmentRetimeConfig(
        min_video_ratio=float(min_video_ratio),
        max_video_ratio=float(max_video_ratio),
        min_audio_tempo=float(min_audio_tempo),
        max_audio_tempo=float(max_audio_tempo),
        keep_gaps=bool(keep_gaps),
        original_audio_volume=float(original_audio_volume),
        crf=int(crf),
        preset=preset,
    )
    audio_paths = [_as_path(item) for item in audio_files]
    for update in retime_video_to_audio_segments(
        _as_path(video_path),
        _as_path(srt_file),
        audio_paths,
        ROOT / "jobs",
        config,
    ):
        yield update.video_path, update.status, update.log_text, update.schedule_path, update.log_path, update.job_path


def batch_retime_srt_audio_segments(batch_job_count, *values):
    sync_values = values[:MAX_BATCH_SYNC_JOBS * 3]
    (
        max_parallel_jobs,
        min_video_ratio,
        max_video_ratio,
        min_audio_tempo,
        max_audio_tempo,
        keep_gaps,
        original_audio_volume,
        crf,
        preset,
    ) = values[MAX_BATCH_SYNC_JOBS * 3:]

    active_count = max(1, min(MAX_BATCH_SYNC_JOBS, int(batch_job_count or 1)))
    video_files = []
    srt_files = []
    audio_segment_groups = []
    for index in range(active_count):
        video_file, srt_file, audio_folder = sync_values[index * 3:index * 3 + 3]
        if not video_file:
            raise gr.Error(f"Bo sync {index + 1}: hay upload video goc.")
        if not srt_file:
            raise gr.Error(f"Bo sync {index + 1}: hay upload file SRT.")
        if not audio_folder:
            raise gr.Error(f"Bo sync {index + 1}: hay nhap folder audio segments.")
        audio_folder_path = Path(str(audio_folder).strip())
        if not audio_folder_path.exists():
            raise gr.Error(f"Bo sync {index + 1}: folder audio segments khong ton tai.")
        if not audio_folder_path.is_dir():
            raise gr.Error(f"Bo sync {index + 1}: duong dan audio segments phai la folder.")
        video_files.append(video_file)
        srt_files.append(srt_file)
        audio_segment_groups.append(audio_folder_path)

    if float(min_video_ratio) > float(max_video_ratio):
        raise gr.Error("Min video ratio phai nho hon hoac bang max video ratio.")
    if float(min_audio_tempo) > float(max_audio_tempo):
        raise gr.Error("Min audio tempo phai nho hon hoac bang max audio tempo.")

    segment_config = SegmentRetimeConfig(
        min_video_ratio=float(min_video_ratio),
        max_video_ratio=float(max_video_ratio),
        min_audio_tempo=float(min_audio_tempo),
        max_audio_tempo=float(max_audio_tempo),
        keep_gaps=bool(keep_gaps),
        original_audio_volume=float(original_audio_volume),
        crf=int(crf),
        preset=preset,
    )
    config = BatchSegmentRetimeConfig(
        segment_config=segment_config,
        max_parallel_jobs=int(max_parallel_jobs),
    )
    video_paths = [_as_path(item) for item in video_files]
    srt_paths = [_as_path(item) for item in srt_files]
    for update in run_batch_segment_retime_uploads(video_paths, srt_paths, audio_segment_groups, ROOT / "jobs", config):
        yield update.status, update.log_text, update.summary_path, update.zip_path, update.batch_job_path


def _batch_job_visibility(count):
    count = max(1, min(MAX_BATCH_SYNC_JOBS, int(count or 1)))
    return [count, *[gr.update(visible=index < count) for index in range(MAX_BATCH_SYNC_JOBS)]]


def add_batch_sync_job(count):
    return _batch_job_visibility((count or 1) + 1)


def remove_batch_sync_job(count):
    return _batch_job_visibility((count or 1) - 1)


with gr.Blocks(title="DichVideo") as demo:
    gr.Markdown("# DichVideo")

    def settings_panel():
        with gr.Row():
            source_language = gr.Dropdown(
                ["auto", "vi", "en", "ja", "ko", "zh", "fr", "de", "es"],
                value="auto",
                label="Ngon ngu goc",
                allow_custom_value=True,
            )
            target_language = gr.Textbox(value="vi", label="Ngon ngu dich")
            whisper_model = gr.Dropdown(
                ["tiny", "base", "small", "medium", "large-v3"],
                value="small",
                label="Whisper model",
            )
            asr_device = gr.Dropdown(
                ["cpu", "cuda", "auto"],
                value="cpu",
                label="ASR device",
            )
            compute_type = gr.Dropdown(
                ["int8", "float16", "float32"],
                value="int8",
                label="Compute type",
            )
        with gr.Row():
            translation_provider = gr.Radio(
                ["none", "deep-translator", "openai"],
                value="none",
                label="Dich",
            )
            openai_model = gr.Textbox(value="gpt-4o-mini", label="OpenAI model")
            tts_voice = gr.Textbox(value="vi-VN-HoaiMyNeural", label="TTS voice")
        with gr.Row():
            chunk_minutes = gr.Slider(1, 30, value=10, step=1, label="Chunk audio (phut)")
            max_tempo = gr.Slider(1.05, 1.80, value=1.35, step=0.05, label="Tang toc audio toi da")
            keep_original_audio = gr.Checkbox(value=False, label="Giu audio goc nho nen")
            original_audio_volume = gr.Slider(0.0, 2.0, value=0.18, step=0.01, label="Am luong audio goc")
            timing_mode = gr.Radio(
                [("Bam sat timestamp", "fit_segments"), ("Khong cat loi doc", "no_cut_sequential")],
                value="fit_segments",
                label="Can timing audio",
            )
            video_when_audio_longer = gr.Radio(
                [("Giu frame cuoi", "hold_last_frame"), ("Keo video khop audio", "stretch_video")],
                value="hold_last_frame",
                label="Khi audio dai hon video",
            )
        return [
            source_language,
            target_language,
            whisper_model,
            asr_device,
            compute_type,
            translation_provider,
            openai_model,
            tts_voice,
            chunk_minutes,
            max_tempo,
            keep_original_audio,
            original_audio_volume,
            timing_mode,
            video_when_audio_longer,
        ]

    with gr.Tabs():
        with gr.Tab("Local pipeline"):
            with gr.Row():
                with gr.Column(scale=3):
                    video_input = gr.Video(label="Video dau vao", sources=["upload"], format=None)
                    run_button = gr.Button("Bat dau local", variant="primary")
                with gr.Column(scale=2):
                    local_settings = settings_panel()

            with gr.Row():
                final_video = gr.Video(label="Video ket qua")
                transcript_file = gr.File(label="Transcript JSON")
                log_file = gr.File(label="Log file")

            status = gr.Textbox(label="Trang thai", lines=4)
            logs = gr.Textbox(label="Log gan day", lines=18, autoscroll=True)

            run_button.click(
                fn=process_video,
                inputs=[video_input, *local_settings],
                outputs=[final_video, status, logs, transcript_file, log_file],
                show_progress="full",
            )

        with gr.Tab("1 ASR + Dich"):
            with gr.Row():
                with gr.Column(scale=3):
                    colab_video_input = gr.Video(label="Video dau vao", sources=["upload"], format=None)
                    drive_jobs_root = gr.Textbox(
                        label="Google Drive jobs folder tren may local",
                        placeholder="G:\\My Drive\\dichvideo\\jobs",
                    )
                    prepare_button = gr.Button("Tao ASR job", variant="primary")
                with gr.Column(scale=2):
                    colab_settings = settings_panel()

            colab_job_path = gr.Textbox(label="Job folder")

            with gr.Row():
                check_button = gr.Button("Kiem tra job")
                translate_button = gr.Button("Dich transcript", variant="primary")
                package_omnivoice_button = gr.Button("Tao OmniVoice input zip", variant="primary")

            with gr.Row():
                colab_transcript_file = gr.File(label="Transcript JSON")
                omnivoice_zip_file = gr.File(label="OmniVoice input zip")
                colab_log_file = gr.File(label="Log file")

            colab_status = gr.Textbox(label="Trang thai", lines=4)
            colab_logs = gr.Textbox(label="Log gan day", lines=18, autoscroll=True)

            prepare_button.click(
                fn=prepare_colab_job,
                inputs=[colab_video_input, drive_jobs_root, *colab_settings],
                outputs=[colab_job_path, colab_status, colab_logs, colab_log_file],
                show_progress="full",
            )
            check_button.click(
                fn=check_job,
                inputs=[colab_job_path],
                outputs=[omnivoice_zip_file, colab_status, colab_logs, colab_transcript_file, colab_log_file],
            )
            translate_button.click(
                fn=translate_asr_job,
                inputs=[colab_job_path, *colab_settings],
                outputs=[colab_status, colab_logs, colab_transcript_file, colab_log_file],
                show_progress="full",
            )
            package_omnivoice_button.click(
                fn=package_omnivoice_job,
                inputs=[colab_job_path],
                outputs=[colab_status, colab_logs, omnivoice_zip_file, colab_log_file],
                show_progress="full",
            )

        with gr.Tab("2 OmniVoice"):
            with gr.Row():
                with gr.Column(scale=3):
                    omnivoice_job_path = gr.Textbox(label="Job folder co output/dubbed.wav")
                    omnivoice_check_button = gr.Button("Kiem tra OmniVoice job")
                    finish_omnivoice_button = gr.Button("Ghep video tu OmniVoice audio", variant="primary")
                with gr.Column(scale=2):
                    omnivoice_settings = settings_panel()

            with gr.Row():
                omnivoice_final_video = gr.Video(label="Video ket qua")
                omnivoice_transcript_file = gr.File(label="Translated JSON")
                omnivoice_log_file = gr.File(label="Log file")

            omnivoice_status = gr.Textbox(label="Trang thai", lines=4)
            omnivoice_logs = gr.Textbox(label="Log gan day", lines=18, autoscroll=True)

            omnivoice_check_button.click(
                fn=check_job,
                inputs=[omnivoice_job_path],
                outputs=[omnivoice_final_video, omnivoice_status, omnivoice_logs, omnivoice_transcript_file, omnivoice_log_file],
            )
            finish_omnivoice_button.click(
                fn=finish_omnivoice_job,
                inputs=[omnivoice_job_path, *omnivoice_settings],
                outputs=[omnivoice_final_video, omnivoice_status, omnivoice_logs, omnivoice_transcript_file, omnivoice_log_file],
                show_progress="full",
            )

        with gr.Tab("3 Sync SRT + Audio Segments"):
            with gr.Row():
                with gr.Column(scale=3):
                    retime_video_input = gr.Video(label="Video goc", sources=["upload"], format=None)
                    retime_srt_input = gr.File(label="SRT goc", file_types=[".srt"], type="filepath")
                    retime_audio_inputs = gr.File(
                        label="Audio segments",
                        file_count="multiple",
                        file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"],
                        type="filepath",
                    )
                    retime_button = gr.Button("Dong bo video theo audio segments", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        min_video_ratio = gr.Slider(0.25, 1.0, value=0.70, step=0.05, label="Video cham toi da")
                        max_video_ratio = gr.Slider(1.0, 3.0, value=1.40, step=0.05, label="Video nhanh toi da")
                    with gr.Row():
                        min_audio_tempo = gr.Slider(0.50, 1.0, value=0.75, step=0.05, label="Audio cham them toi da")
                        max_audio_tempo = gr.Slider(1.0, 2.0, value=1.80, step=0.05, label="Audio nhanh them toi da")
                    with gr.Row():
                        keep_gaps = gr.Checkbox(value=True, label="Giu khoang lang giua cac subtitle")
                        retime_original_audio_volume = gr.Slider(0.0, 2.0, value=0.0, step=0.01, label="Am luong audio goc")
                        crf = gr.Slider(16, 30, value=20, step=1, label="CRF video")
                        preset = gr.Dropdown(
                            ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
                            value="veryfast",
                            label="FFmpeg preset",
                        )

            with gr.Row():
                retime_final_video = gr.Video(label="Video ket qua")
                retime_schedule_file = gr.File(label="Retime schedule JSON")
                retime_log_file = gr.File(label="Log file")

            retime_job_path = gr.Textbox(label="Job folder", lines=1)
            retime_status = gr.Textbox(label="Trang thai", lines=4)
            retime_logs = gr.Textbox(label="Log gan day", lines=18, autoscroll=True)

            retime_button.click(
                fn=retime_srt_audio_segments,
                inputs=[
                    retime_video_input,
                    retime_srt_input,
                    retime_audio_inputs,
                    min_video_ratio,
                    max_video_ratio,
                    min_audio_tempo,
                    max_audio_tempo,
                    keep_gaps,
                    retime_original_audio_volume,
                    crf,
                    preset,
                ],
                outputs=[
                    retime_final_video,
                    retime_status,
                    retime_logs,
                    retime_schedule_file,
                    retime_log_file,
                    retime_job_path,
                ],
                show_progress="full",
            )

        with gr.Tab("4 Batch Sync Parallel"):
            batch_job_count = gr.State(1)
            with gr.Row():
                with gr.Column(scale=3):
                    batch_sync_groups = []
                    batch_sync_inputs = []
                    for index in range(MAX_BATCH_SYNC_JOBS):
                        with gr.Group(visible=index == 0) as batch_sync_group:
                            gr.Markdown(f"### Bo sync {index + 1}")
                            batch_video_input = gr.File(
                                label="Video goc",
                                file_count="single",
                                file_types=[".mp4", ".mov", ".mkv", ".webm", ".avi"],
                                type="filepath",
                            )
                            batch_srt_input = gr.File(
                                label="SRT goc",
                                file_count="single",
                                file_types=[".srt"],
                                type="filepath",
                            )
                            batch_audio_input = gr.Textbox(
                                label="Folder audio segments",
                                placeholder=r"D:\audio_results\video_001\segments",
                            )
                        batch_sync_groups.append(batch_sync_group)
                        batch_sync_inputs.extend([batch_video_input, batch_srt_input, batch_audio_input])
                    with gr.Row():
                        add_batch_job_button = gr.Button("Them bo sync")
                        remove_batch_job_button = gr.Button("Xoa bo sync cuoi")
                    batch_retime_button = gr.Button("Chay batch sync song song", variant="primary")
                with gr.Column(scale=2):
                    batch_max_parallel_jobs = gr.Slider(
                        1,
                        4,
                        value=2,
                        step=1,
                        label="So job chay song song",
                    )
                    with gr.Row():
                        batch_min_video_ratio = gr.Slider(0.25, 1.0, value=0.70, step=0.05, label="Video cham toi da")
                        batch_max_video_ratio = gr.Slider(1.0, 3.0, value=1.40, step=0.05, label="Video nhanh toi da")
                    with gr.Row():
                        batch_min_audio_tempo = gr.Slider(0.50, 1.0, value=0.75, step=0.05, label="Audio cham them toi da")
                        batch_max_audio_tempo = gr.Slider(1.0, 2.0, value=1.80, step=0.05, label="Audio nhanh them toi da")
                    with gr.Row():
                        batch_keep_gaps = gr.Checkbox(value=True, label="Giu khoang lang giua cac subtitle")
                        batch_original_audio_volume = gr.Slider(0.0, 2.0, value=0.0, step=0.01, label="Am luong audio goc")
                        batch_crf = gr.Slider(16, 30, value=20, step=1, label="CRF video")
                        batch_preset = gr.Dropdown(
                            ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
                            value="veryfast",
                            label="FFmpeg preset",
                        )

            gr.Markdown(
                "Moi bo sync chi nhan 1 video, 1 SRT, va 1 duong dan folder audio segments. Bam them/xoa bo sync de chay nhieu video cung luc."
            )
            with gr.Row():
                batch_summary_file = gr.File(label="Batch summary JSON")
                batch_zip_file = gr.File(label="Final videos ZIP")

            batch_job_path = gr.Textbox(label="Batch job folder", lines=1)
            batch_status = gr.Textbox(label="Trang thai", lines=4)
            batch_logs = gr.Textbox(label="Log gan day", lines=18, autoscroll=True)

            add_batch_job_button.click(
                fn=add_batch_sync_job,
                inputs=[batch_job_count],
                outputs=[batch_job_count, *batch_sync_groups],
            )
            remove_batch_job_button.click(
                fn=remove_batch_sync_job,
                inputs=[batch_job_count],
                outputs=[batch_job_count, *batch_sync_groups],
            )
            batch_retime_button.click(
                fn=batch_retime_srt_audio_segments,
                inputs=[
                    batch_job_count,
                    *batch_sync_inputs,
                    batch_max_parallel_jobs,
                    batch_min_video_ratio,
                    batch_max_video_ratio,
                    batch_min_audio_tempo,
                    batch_max_audio_tempo,
                    batch_keep_gaps,
                    batch_original_audio_volume,
                    batch_crf,
                    batch_preset,
                ],
                outputs=[
                    batch_status,
                    batch_logs,
                    batch_summary_file,
                    batch_zip_file,
                    batch_job_path,
                ],
                show_progress="full",
            )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
