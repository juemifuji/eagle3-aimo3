export PYTHONPATH=/path/eagle3/SpecForge:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# ====== DeepSpeed multi-node config ======
NUM_NODES=1
GPUS_PER_NODE=8

HOSTFILE=${HOSTFILE:-$ROOT_DIR/hostfile_2n8g}

BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-128}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 可选：提升 NCCL 稳定性（按你们集群网卡情况调整/删除）
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NNODES=1
export NODE_RANK=$1
export GPUS_PER_NODE=8

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --node_rank=$1 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  "$ROOT_DIR/scripts/train_eagle3.py" \
    --target-model-path /path/gpt-oss-120b \
    --draft-model-config "$ROOT_DIR/configs/gpt-oss-120B-eagle3.json" \
    --train-data-path /path/data/train.json \
    --eval-data-path /path/data/test.json \
    --build-dataset-num-proc "$BUILD_DATASET_NUM_PROC" \
    --output-dir "$ROOT_DIR/models/gpt-oss-120b-eagle3" \
    --tp-size 8 \
    --target-model-backend sglang \
    --sglang-attention-backend fa3 \
    --num-epochs 10 \
    --batch-size 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --learning-rate 1e-4 \
    --max-length 45056 \
    --distill-mode hard \
    --logits-chunk-size 128 \
    --ttt-length 3 \
    --chat-template gpt-oss \
    --is-preformatted \
    --cache-dir "$ROOT_DIR/cache" \
    --dist-timeout 60
