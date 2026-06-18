#!/bin/bash
# Train the DiT (cross-attention over frozen-VFM tokens) policy.
# Args: task_name task_config expert_data_num seed action_dim gpu_id [encoder]
# Run from policy/DiT (like DP's train.sh is run from policy/DP).

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}
enc=${7:-dinov3_ss}

head_camera_type=D435
config_name=robot_dit_feat_${action_dim}

# token grid (N, C) per encoder
case $enc in
  dinov3_ss|dinov3_ln) tok_n=196;  tok_c=1024 ;;
  clip_ss)             tok_n=256;  tok_c=1024 ;;
  depth_ss|depth_v2_ss) tok_n=1369; tok_c=1024 ;;
  sam_ss)              tok_n=4096; tok_c=256  ;;
  vjepa_ss)            tok_n=256;  tok_c=1024 ;;
  *) echo "unknown encoder $enc"; exit 1 ;;
esac

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
export HF_HOME=${HF_HOME:-$(cd ../../../.. && pwd)/.cache/huggingface}
export HF_HUB_OFFLINE=1

DIT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$DIT_DIR/../.." && pwd)
cd "$ROOT"

raw="policy/DP/data/${task_name}-${task_config}-${expert_data_num}.zarr"
tokz="policy/DP/data/feat_tok/${enc}/${task_name}-${task_config}-${expert_data_num}.tokzarr"

if [ ! -d "$raw" ]; then
  echo "raw zarr missing: $raw (run DP process_data first)"; exit 1
fi
if [ ! -d "$tokz" ]; then
  echo "precomputing tokens -> $tokz"
  python policy/DiT/precompute_tokens.py ${task_name} ${task_config} ${expert_data_num} ${enc}
fi

python policy/DiT/train.py --config-name=${config_name}.yaml \
    task.name=${task_name} \
    task.dataset.zarr_path="data/feat_tok/${enc}/${task_name}-${task_config}-${expert_data_num}.tokzarr" \
    tok_n=${tok_n} tok_c=${tok_c} encoder_tag=${enc}_dit \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${task_name}-dit-${enc} \
    logging.mode=offline \
    setting=${task_config} \
    expert_data_num=${expert_data_num} \
    head_camera_type=$head_camera_type
