### EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test

[gpt-oss-120b-eagle3-aimo3](https://huggingface.co/wenliang1990/gpt-oss-120b-eagle3-aimo3)

Our code repository is a secondary development based on [SpecForge](https://github.com/sgl-project/SpecForge). We optimized GPU memory usage, enabling **gpt-oss-120b** to train with context lengths beyond **40K** on 8×H800 GPUs. The original repository could not reliably support long-context training on eight H800s—especially for gpt-oss-120b—where the maximum context length was typically around 16K. With our optimizations, the trainable context length is extended to 40K+, and training throughput is significantly improved.

| File Name                                          | Original Implementation                                                                                                              | New Implementation                                                                                                                   | Key Benefit                                                                                           |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `specforge/core/eagle.py`                          | Soft distillation: slice teacher logits from `[B, T, V]` to `[B, T, Vd]`, then apply Softmax to obtain `target_p` (3D distribution). | Optional hard distillation (recommended): use only the teacher’s top-1 token to produce `labels` `[B, T]` (2D) plus `position_mask`. | Target tensor size reduced from **O(B·T·Vd)** to **O(B·T)**, significantly lowering GPU memory usage. |
| `specforge/data/parse.py`                          | Python tool outputs are included in loss computation.                                                                                | Python tool outputs are excluded from loss computation.                                                                              | Avoids optimizing on tool-generated outputs (more stable/cleaner training signal).                    |
| `specforge/modeling/target/eagle3_target_model.py` | Targets store teacher logits `[B, T, V]` (SGLang directly passes logits; HF/Custom also return logits).                              | Targets store teacher top-1 token IDs `[B, T]` (int64); HF/Custom/SGLang all convert logits to argmax token IDs.                     | Memory reduced from **O(B·T·V)** to **O(B·T)**.                                                       |


### Evaluation
We provide a test script to evaluate the acceleration ratio. The test data is sourced from the [International Mathematical Olympiad (IMO)](https://huggingface.co/datasets/Hwilner/imo-answerbench), and the testing environment uses NVIDIA H800 GPUs.
To run the benchmark:
```text
python benchmark.py
```

| throughput(gpt-oss-120b) | throughput(gpt-oss-120b-eagle3-aimo3) | speedup | concurrency |
| ------------------------ | ------------------------------------ | ------- | ----------- |
| 776.514                  | 1059.43                              | 36.40%  | 8           |
| 686.717                  | 956.431                              | 39.30%  | 7           |
| 596.596                  | 851.647                              | 42.80%  | 6           |
| 518.76                   | 680.951                              | 31.30%  | 5           |
| 465.702                  | 657.682                              | 41.20%  | 4           |
| 379.48                   | 541.304                              | 42.60%  | 3           |
| 297.553                  | 422.232                              | 41.90%  | 2           |
| 190.023                  | 268.132                              | 41.10%  | 1           |

### Serving

```text
TP="${TP:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-40960}"
STREAM_INTERVAL="${STREAM_INTERVAL:-1}"

# ====== speculative config (JSON) ======
SPECULATIVE_CONFIG='{"method":"eagle3","model":"gpt-oss-120b-eagle3-aimo3","num_speculative_tokens":3,"draft_tensor_parallel_size":1}'

exec python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --served-model-name gpt-oss \
  --tensor-parallel-size "$TP" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --max-model-len "$MAX_LEN" \
  --async-scheduling \
  --stream-interval "$STREAM_INTERVAL" \
  --speculative-config "$SPECULATIVE_CONFIG"
```

### Example Usage

```text
python test.py \
  --model-path /path/gpt-oss-120b \
  --served-model-name gpt-oss \
  --port 8001 \
  --base-url-host 127.0.0.1 \
  --speculative-config '{"method":"eagle3","model":"/path/gpt-oss-120b-eagle3-aimo3","num_speculative_tokens":3,"draft_tensor_parallel_size":1}' \
  --query "3424*4334+342342+943"
```
