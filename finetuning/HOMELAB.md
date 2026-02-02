# Qwen3-TTS Finetuning on Homelab

**Repo**: https://github.com/thinhlpg/Qwen3-TTS (private)  
**Homelab path**: `~/code/Qwen3-TTS/`  
**Meshnet**: `100.80.53.250`

## Environment

- **Python**: 3.12+ (use repo `.venv`)
- **GPU**: RTX 5090 (32GB), use `CUDA_VISIBLE_DEVICES=0` or `1`
- **Data**: `~/data/vivoice/` (viVoice parquets), `~/datasets/<speaker>/` (extracted)
- **Checkpoints**: `~/checkpoints/qwen3-tts-1.7b-<speaker>/`

## 1. viVoice Parquets

- **Location**: `~/data/viVoice/data/` (354 parquet files, ~157GB)
- **Download** (if missing): `python ~/code/my-dataset-catalog/scripts/download_vivoice.py` (from Qwen3-TTS venv)
- **Columns**: `channel`, `text`, `audio` (WAV bytes in `audio["bytes"]`)

## 2. Extract Speaker Dataset

Channel names in parquets are YouTube handles (e.g. `@CoBaBinhDuong`, `@hieu-tv`).

```bash
cd ~/code/Qwen3-TTS && source .venv/bin/activate
python /tmp/extract_cobabinhduong.py   # or use your extraction script
```

**Output**: `~/datasets/<speaker>/train_raw.jsonl` and `~/datasets/<speaker>/wavs/*.wav`

## 3. Prepare Audio Codes (GPU 0)

```bash
cd ~/code/Qwen3-TTS && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl /home/thinhlpg/datasets/<speaker>/train_raw.jsonl \
  --output_jsonl /home/thinhlpg/datasets/<speaker>/train_with_codes.jsonl
```

## 4. Train (GPU 0)

```bash
cd ~/code/Qwen3-TTS && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -u finetuning/sft_12hz_mlflow.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl /home/thinhlpg/datasets/<speaker>/train_with_codes.jsonl \
  --output_model_path /home/thinhlpg/checkpoints/qwen3-tts-1.7b-<speaker> \
  --batch_size 4 \
  --num_epochs 20 \
  --lr 2e-6 \
  --speaker_name <speaker>
```

**Example (cobabinhduong)**: 4,000 samples, batch 4, 20 epochs, speaker_name `cobabinhduong`.

## 5. ComfyUI Model

Copy best checkpoint to ComfyUI:

```bash
cp -r ~/checkpoints/qwen3-tts-1.7b-<speaker>/checkpoint-epoch-<N> \
      ~/checkpoints/comfyui-models/Qwen3-TTS/<speaker>-1.7b-epoch<N>/
```

## Key Paths

| What | Path |
|------|------|
| Repo | `~/code/Qwen3-TTS` |
| viVoice parquets | `~/data/viVoice/data/` |
| Extracted dataset | `~/datasets/<speaker>/` |
| Train JSONL (raw) | `~/datasets/<speaker>/train_raw.jsonl` |
| Train JSONL (with codes) | `~/datasets/<speaker>/train_with_codes.jsonl` |
| Checkpoints | `~/checkpoints/qwen3-tts-1.7b-<speaker>/` |
| ComfyUI Qwen3-TTS | `~/checkpoints/comfyui-models/Qwen3-TTS/` |
| MLflow | http://mlflow.thinhlpg.me |

## Verified Training Runs

| Speaker | Samples | Epochs | Batch | Best Epoch | Checkpoint Path |
|---------|---------|--------|-------|------------|-----------------|
| hieutv | 2,000 | 33 | 2 | 13 | `~/checkpoints/qwen3-tts-1.7b-hieutv/` |
| cobabinhduong | 4,000 | 20 | 4 | TBD | `~/checkpoints/qwen3-tts-1.7b-cobabinhduong/` |
