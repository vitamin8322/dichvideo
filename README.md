# DichVideo

Local Gradio tool for video dubbing:

1. Extract audio from a video with FFmpeg.
2. Transcribe speech locally with faster-whisper.
3. Translate segments.
4. Generate TTS audio.
5. Fit translated audio to original segment timing.
6. Mux dubbed audio back into the original video.

`Am luong audio goc` controls the original video's audio when you keep/mix it:

- `0`: mute/remove original audio.
- `0.18`: small background audio.
- `1.0`: original volume.
- `2.0`: double volume.

## Setup

Install FFmpeg first and make sure `ffmpeg` and `ffprobe` are available in PATH.

```powershell
py -m pip install -r requirements.txt
```

Run the UI:

```powershell
py app.py
```

Open the Gradio URL printed in the terminal.

## Translation Providers

- `none`: keeps the ASR text as-is. Useful for testing the pipeline.
- `deep-translator`: uses Google Translate through `deep-translator`.
- `openai`: requires `OPENAI_API_KEY` in your environment.

## ASR Device

The UI defaults to `cpu` because it works on most Windows machines. If you have a working CUDA setup, switch `ASR device` to `cuda` and use `float16` or `int8`.

## Part 1: ASR + Translation

Use this when you want local video processing but ASR runs on a free Colab GPU.

1. Install Google Drive for Desktop.
2. Create a Drive folder such as:

```text
G:\My Drive\dichvideo\jobs
```

3. In the Gradio UI, open `1 ASR + Dich`.
4. Select a video and set `Google Drive jobs folder tren may local` to that folder.
5. Click `Tao job cho Colab`.
6. The app creates:

```text
G:\My Drive\dichvideo\jobs\job_YYYYMMDD_HHMMSS
G:\My Drive\dichvideo\colab_asr_worker.py
```

7. Open `notebooks/dichvideo_colab_worker.ipynb` in Google Colab.
8. Mount Drive, set `JOB_DIR`, and run the worker cell.
9. When Colab writes `output/transcript.json`, return to Gradio and click `Dich transcript`.
10. Click `Tao OmniVoice input zip` if you want cloned-voice TTS on Colab.

Colab worker outputs:

```text
output/transcript.json
output/transcript_chunk_0000.json
logs/colab_asr.log
status.json
```

Example:

```powershell
$env:OPENAI_API_KEY="..."
py app.py
```

## Logs

Every run creates a folder under `jobs/`, for example:

```text
jobs/job_20260702_191501/
  logs/pipeline.log
  manifest.json
  output/transcript.json
  output/translated.json
  output/final.mp4
```

If something breaks, send `logs/pipeline.log`, `manifest.json`, and the latest terminal output.
For Colab ASR bugs, also send `logs/colab_asr.log` and `status.json`.

## Batch Colab ASR To SRT

`notebooks/dichvideo_colab_worker.ipynb` now runs as a standalone Colab batch notebook:

1. Upload one or more video/audio files.
2. Colab runs `faster-whisper`.
3. It downloads `asr_srt_results.zip`.

The zip contains only:

```text
*.srt
batch_asr_to_srt.log
```

## Batch Colab OmniVoice From SRT

`notebooks/dichvideo_omnivoice_tts_worker.ipynb` now runs as a standalone Colab batch notebook:

1. Upload one or more `.srt` files.
2. Upload one reference voice audio, such as `audio-truyen.mp3`.
3. Upload `scripts/colab_batch_omnivoice_from_srt.py`.
4. Colab runs OmniVoice and downloads `omnivoice_audio_results.zip`.

Important: OmniVoice reads the text inside the SRT files. If you want Vietnamese audio, upload Vietnamese translated SRT files.

For each SRT, the output contains:

```text
<srt-name>/
  <srt-name>_full.wav
  <srt-name>.srt
  timing_schedule.json
  segments/
    0001.wav
    0002.wav
    ...
  mix_segments/
    0001.wav
    0002.wav
    ...
```

## Batch Colab F5-TTS Vietnamese From SRT

Use this when OmniVoice/VoxCPM2 drops short tokens such as `cap S`.

`notebooks/10_f5tts_vietnamese_colab.ipynb` runs `hynt/F5-TTS-Vietnamese-ViVoice` on Colab:

1. Upload one or more Vietnamese `.srt` files.
2. Upload one reference voice audio, such as `audio-truyen.mp3`.
3. Upload `scripts/colab_batch_f5tts_vietnamese_from_srt.py`.
4. Colab runs F5-TTS and downloads `f5tts_vietnamese_audio_results.zip`.

The script normalizes risky short tokens before TTS, for example `cap S`/`cấp S` is spoken as `cap et`/`cấp et`, and `SSS` is expanded letter by letter. Pass `--disable-text-normalize` if you want the raw SRT text.

For each SRT, the output folder shape matches the OmniVoice batch output:

```text
<srt-name>/
  <srt-name>_full.wav
  <srt-name>.srt
  timing_schedule.json
  segments/
  mix_segments/
```

## Sync Video With SRT + Audio Segments

Use tab `3 Sync SRT + Audio Segments` after you already have:

- the original video,
- the original SRT,
- one generated audio file per SRT segment.

The tool compares each SRT segment duration with the matching audio segment duration. It first retimes the video segment to match the audio. If the required video speed is outside your configured range, it clamps the video speed and then adjusts the audio tempo just enough to fit.

Set `Am luong audio goc` above `0` if you want to keep the original video's sound/music underneath the generated speech. In this sync tab, the original audio is retimed per segment together with the video before mixing, so it stays aligned with the retimed picture.

Outputs are written under:

```text
jobs/segment_retime_YYYYMMDD_HHMMSS/
  output/final.mp4
  output/mixed_audio.wav              # only when original audio volume > 0
  output/original_retimed_audio.wav   # only when original audio volume > 0
  output/retime_schedule.json
  logs/segment_retimer.log
```

If sync looks wrong, send `output/retime_schedule.json` and `logs/segment_retimer.log`.

## Batch Sync Video With SRT + Audio Segments

Use tab `4 Batch Sync Parallel` when you want to create many synced videos in one run.

Add one `Bo sync` per video. Each set has exactly one video, one SRT file, and one audio segment ZIP:

```text
Bo sync 1:
  video 1
  srt 1
  audio_zip 1

Bo sync 2:
  video 2
  srt 2
  audio_zip 2
```

Each audio ZIP should contain the segment files for one video:

```text
0001.wav
0002.wav
0003.wav
...
```

The ZIP may also contain those files inside a folder; the tool extracts supported audio files and ignores non-audio files.

Set `So job chay song song` conservatively. `2` is a good default; higher values use more CPU/GPU/FFmpeg resources at the same time.

Outputs are written under:

```text
jobs/batch_segment_retime_YYYYMMDD_HHMMSS/
  output/batch_summary.json
  output/batch_final_videos.zip
  output/final_videos/
  output/schedules/
  output/logs/
  logs/batch_segment_retimer.log
  jobs/segment_retime_.../
```

Uploaded inputs are staged under `jobs/batch_upload_staging_YYYYMMDD_HHMMSS/`.

If one video fails, the batch continues with the remaining videos. Send `output/batch_summary.json`, `logs/batch_segment_retimer.log`, and the failed child job log when you need bug review.

## Part 2: Colab OmniVoice TTS

Use this after Part 1 has produced `output/translated.json` and an OmniVoice input zip.

Manual upload flow:

1. Upload these files to Colab:

```text
job_xxx_omnivoice_input.zip
audio-truyen.mp3
scripts/colab_omnivoice_tts_worker.py
```

2. Run `notebooks/dichvideo_omnivoice_tts_worker.ipynb`.
3. Download `job_xxx_omnivoice_result.zip`.
4. Extract it on the local machine.
5. In the UI, open `2 OmniVoice`.
6. Enter the extracted folder as `Job folder co output/dubbed.wav`, then click `Ghep video tu OmniVoice audio`.

OmniVoice outputs:

```text
output/dubbed.wav
output/timing_schedule.json
logs/omnivoice_tts.log
status.json
```
