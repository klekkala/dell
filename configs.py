from arguments import get_args
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback
import os

args = get_args()




resource_file = '/lab/kiran/hostconf/'

all_envs = ["AirRaidNoFrameskip-v4","AssaultNoFrameskip-v4","BeamRiderNoFrameskip-v4", "CarnivalNoFrameskip-v4","DemonAttackNoFrameskip-v4","NameThisGameNoFrameskip-v4","PhoenixNoFrameskip-v4","RiverraidNoFrameskip-v4","SpaceInvadersNoFrameskip-v4"]

if args.backbone == "e2e":
    args.train_backbone = True

atari_config = {
    "env" : args.env_name,
    "clip_rewards" : True,
    "framework" : "torch",
    "logger_config": {
        "type": UnifiedLogger,
        "logdir": os.path.expanduser(args.log)
        },
    "observation_filter":"NoFilter",
    "num_workers": args.num_workers,
    "rollout_fragment_length" : 100,
    "num_envs_per_worker" : args.num_envs,
    "model": {
        "custom_model": "model",
        "vf_share_layers": True,
        "conv_filters": [[16, [8, 8], 4], [32, [4, 4], 2], [512, [11, 11], 1],],
        "conv_activation" : "relu" if (args.temporal == '4stack' or args.temporal == 'notemp') else "elu",
        "custom_model_config" : {"backbone": args.backbone, "backbone_path": args.ckpt + args.env_name + "/" + args.backbone, "train_backbone": args.train_backbone, 'temporal': args.temporal, "div": args.div},
        "framestack": args.temporal == '4stack',
        "use_lstm": args.temporal == 'lstm',
        "use_attention": args.temporal == 'attention',
    },
    "horizon": 4650,
    "kl_coeff" : args.kl_coeff,
    "clip_param" : args.clip_param,
    "entropy_coeff" : args.entropy_coeff,
    "gamma" : args.gamma,
    "lr" : args.lr,
    "vf_clip_param" : args.vf_clip,
    "train_batch_size":args.buffer_size,
    "sgd_minibatch_size":args.batch_size,
    "num_sgd_iter":args.num_epoch,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus":args.num_gpus,
    "num_gpus_per_worker" : args.gpus_worker,
    "num_cpus_per_worker":args.cpus_worker
    }

