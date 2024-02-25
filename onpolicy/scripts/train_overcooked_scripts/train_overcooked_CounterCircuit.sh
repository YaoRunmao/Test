#!/bin/sh
env="Overcooked"
scenario="Counter Circuit"
algo="mappo"
exp="check"
seed=1

num_agents=2

# train param
num_env_steps=300000000
episode_length=400

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}"

CUDA_VISIBLE_DEVICES=0 python ../train/train_overcooked.py \
--env_name ${env} --scenario_name "${scenario}" --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--n_rollout_threads 100 --ppo_epoch 10 --num_mini_batch 2 --lr 7e-4 --critic_lr 7e-4 --clip_param 0.15 \
--wandb_name "xxx" --user_name "yrm21"
