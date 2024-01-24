import argparse
import torch

#(args.backbone, args.setting, args.trainset, args.shared)
#args.expname only makes sense if args.setting is not allgame

#if args.backbone e2e.. then the gradients flow through the backbone during training
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    #parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        #choices=["e2e", "vae", "alloff", "eachmixedoff", "eachmediumoff", "eachexpertoff", "allmixedoff", "allmediumoff", "allexpertoff", "random", "value", "1channel_vae", "4stack_vae"],
        default="e2e",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
             "be achieved within --stop-timesteps AND --stop-iters.",
    )
    
    parser.add_argument(
        "--machine", type=str, default="", help="machine to be training"
    )
    parser.add_argument(
        "--log", type=str, default="/lab/kiran/logs/rllib", help="config file for resources"
    )

    parser.add_argument(
        "--ckpt", type=str, default="/lab/kiran/ckpts/pretrained/", help="directory for saving resources"
    ) 
    parser.add_argument(
        "--env_name", type=str, default="atari", help="Environment name"
    )
    parser.add_argument(
        "--set", type=str, default="all", help="ALE/Pong-v5"
    )
    parser.add_argument(
        "--data_path", type=str, default="", help="for beogym"
    )
    parser.add_argument(
        "--shared", type=str, choices=["backbone", "backbonepolicy", "full", "ours"], default="full", help="ALE/Pong-v5"
    )
    parser.add_argument(
        "--policy", type=str, default="PolicyNotLoaded", help="ALE/Pong-v5"
    )    
    parser.add_argument(
        "--temporal", type=str, choices=["attention", "lstm", "4stack", "notemp"], default="4stack", help="temporal model"
    )
    parser.add_argument(
        "--prefix", type=str, default="1.a", help="which baseline is it"
    )
    parser.add_argument(
        "--train", action='store_true'
    )
    parser.add_argument(
        "--train_backbone", action='store_true'
    )
    parser.add_argument(
        "--eval", action='store_true'
    )
    parser.add_argument(
        "--pbt", default=False, action='store_true'
    )
    parser.add_argument(
        "--stop_timesteps", type=int, default=25000000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--div", type=float, default=1.0, help="Dividing by 1.0 or 255.0"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--kl_coeff", type=float, default=0.0, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--clip_param", type=float, default=.1, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--entropy_coeff", type=float, default=.01, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--gamma", type=float, default=.95, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--vf_clip", type=float, default=10.0, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--buffer_size", type=int, default=20000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--num_epoch", type=int, default=10, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--num_workers", type=int, default=9, help="Number of GPUs each worker has"
    )
    
    parser.add_argument(
        "--num_envs", type=int, default=5, help="Number of envs each worker evaluates"
    )
    
    parser.add_argument(
        "--num_gpus", type=float, default=.4, help="Number of GPUs each worker has"
    )

    parser.add_argument(
        "--gpus_worker", type=float, default=.2, help="Number of GPUs each worker has"
    ) 

    parser.add_argument(
        "--cpus_worker", type=float, default=1, help="Number of CPUs each worker has"
    )

    #use_lstm or framestacking
    parser.add_argument(
        "--no_tune",
        action="store_true",
        default=True,
        help="Run with/without Tune using a manual train loop instead. If ran without tune, use PPO without grid search and no TensorBoard.",
    )

    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    args = parser.parse_args()

    return args
