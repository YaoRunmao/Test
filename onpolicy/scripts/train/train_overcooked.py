#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
import torch
import gym
from pathlib import Path
from onpolicy.config import get_config
from onpolicy.envs.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from onpolicy.envs.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

"""Train script for Overcooked."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.scenario_name == "Cramped Room":
                    mdp = OvercookedGridworld.from_layout_name("cramped_room")
                elif all_args.scenario_name == "Asymmetric Advantages":
                    mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
                elif all_args.scenario_name == "Coordination Ring":
                    mdp = OvercookedGridworld.from_layout_name("coordination_ring")
                elif all_args.scenario_name == "Forced Coordination":
                    mdp = OvercookedGridworld.from_layout_name("forced_coordination")
                elif all_args.scenario_name == "Counter Circuit":
                    mdp = OvercookedGridworld.from_layout_name("random3")
                else:
                    print("Can not support the " +
                          all_args.scenario_name + "scenario.")
                    raise NotImplementedError

                base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
                env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp, seed=all_args.seed + rank * 1000)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.scenario_name == "Cramped Room":
                    mdp = OvercookedGridworld.from_layout_name("cramped_room")
                elif all_args.scenario_name == "Asymmetric Advantages":
                    mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
                elif all_args.scenario_name == "Coordination Ring":
                    mdp = OvercookedGridworld.from_layout_name("coordination_ring")
                elif all_args.scenario_name == "Forced Coordination":
                    mdp = OvercookedGridworld.from_layout_name("forced_coordination")
                elif all_args.scenario_name == "Counter Circuit":
                    mdp = OvercookedGridworld.from_layout_name("random3")
                else:
                    print("Can not support the " +
                          all_args.scenario_name + "scenario.")
                    raise NotImplementedError

                base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
                env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp, seed=all_args.seed * 50000 + rank * 10000)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(
                             all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
        all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    from onpolicy.runner.shared.overcooked_runner import OvercookedRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
