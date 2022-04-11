#!/bin/sh
path_A="model_dialogpt_fc_lastonly_0.1ga_10KL_1attack_shuffle_newext"
path_B="model_dialogpt_fc_lastonly_0.1ga_10KL_1attack_shuffle_newext"
external_path="attackers/attacker_2layer_0.1ga_10KL_1attack_shuffle_newext"
generate_path=$path_A".json"
save_dir=$external_path".json"
# attcker should be '2layer' or 'trans'
attacker="2layer"
eval "$(conda shell.bash hook)"
conda activate pytorch
# test attacker
CUDA_VISIBLE_DEVICES=3 python eval_privacy.py  $path_A  $path_B  $external_path  $save_dir $attacker
# test generation
CUDA_VISIBLE_DEVICES=3 python generate.py  $path_A   $path_B  $generate_path
#conda activate bert_score
CUDA_VISIBLE_DEVICES=3 python  metrics.py  $generate_path $external_path
