from easydict import EasyDict

import torch
from torch.utils.tensorboard import SummaryWriter
import random
import time

time_str = time.strftime('%d_%H_%M_%S')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = dict(
    selfplay=True,
    gamma=0.99,
    device = device,
    greedy_init=0.5,
    greedy_min=0.1,
    greedy_decrease=0.01,
    model=dict(  # 训练配置项
        device=device,
        learning_rate=0.001,
        critic_learning_rate=0.001,
        training_times=15,
        batch_size=512,
        clip_epsilon=0.2,
        entropy_coef=0.018,
        value_loss_coef=1.0,
        target_kl=None,
        actor_max_grad_norm=5,
        critic_max_grad_norm=5,
        load_from=None
    ),
    critic=dict(
        lr = 0.03,
        save_path='./checkpoints/critic_ppo.pth',
        # load_from='./checkpoints/critic_ppo.pth',
        load_from=None
    ),
    actor=dict(
        lr = 0.03,
        save_path='./checkpoints/actor_ppo.pth',
        # load_from='./checkpoints/actor_ppo.pth',
        load_from=None
    ),
    ppo=dict(
        batch_learn_times=50,
        eps_clip=0.2,
        has_coninuous_action_space=False,
        K_epochs = 5,
        use_gae=False,
        gae_lambda=0.95,

    ),
    ddpg=dict(
        soft_update_x = 0.9,
        buffer_capacity = 2048,
        batch_size = 512,
    ),
    env_cfg=dict(
        name="LunarLander-v2",
        state_dim=0,
        act_dim = 0
    ),
    train=dict(
        max_episodes=int(3e6),
        max_episodes_length = 1000,
        save_model_freq= int(1e3), # per X episodes
        batch_size=2048,
        render_freq = int(100),
    ),
    logs=dict(
        tensorboard=None,
        # tensorboard=None,
        reward_write_freq = 10, # write per X epoch  
    ),
    # for multiStepLr
    warmup=dict(
        gamma = 0.1,
        milestones=[20,80]
    )
)
config = EasyDict(config)
