#!/bin/bash
# Eval the DiT policy. Args: task_name task_config ckpt_setting expert_data_num seed gpu_id [encoder]
policy_name=DiT
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
enc=${7:-dinov3}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HF_HUB_OFFLINE=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --encoder_tag ${enc}_dit \
    --feat_encoder ${enc}
