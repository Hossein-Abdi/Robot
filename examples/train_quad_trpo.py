import gymnasium as gym
import os
import sys
import time
from typing import Any, Tuple
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
sys.path.append("algorithm")
from algorithm.init import *
import algorithm.fcn as fcn
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
# import the skrl components to build the RL system
from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.wrappers.torch import GymnasiumWrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)
from gym_quadruped.quadruped_env import QuadrupedEnv
import copy
import config.config_aliengo as config

class OurGymnasiumWrapper(GymnasiumWrapper):
    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        # handle vectorized environments (vector environments are autoreset)
        if self._vectorized:
            if self._reset_once:
                observation, self._info = self._env.reset()
                self._observation = flatten_tensorized_space(
                    tensorize_space(self.observation_space, observation, device=self.device)
                )
                self._reset_once = False
            return self._observation, self._info

        observation = self._env.reset()
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, device=self.device))
        return observation, None

class RewQuadrupedEnv(QuadrupedEnv):
    def _compute_reward(self):
        lin_rew = np.sum(np.exp(-self.base_lin_vel_err()))
        ang_rew = np.sum(np.exp(-self.base_ang_vel_err()))
        torque_penalty = 0.1 * (np.sum(self.mjData.ctrl >= config.max_torque) + np.sum(self.mjData.ctrl <= config.min_torque))
        work_penalty = 0.01 * self.work
        # 10*(self.mjData.qpos[0] - 1.)
        # print(lin_rew, ang_rew, torque_penalty, work_penalty)
        return lin_rew+ang_rew-torque_penalty-work_penalty


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 256)
        self.linear_layer_2 = nn.Linear(256, 256)
        self.action_layer = nn.Linear(256, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        # Pendulum-v1 action_space is -2 to 2
        return torch.tanh(self.action_layer(x)), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 256)
        self.linear_layer_2 = nn.Linear(256, 256)
        self.linear_layer_3 = nn.Linear(256, 1)

    def compute(self, inputs, role):
        # print(inputs["states"].shape)
        # pdb.set_trace()
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}

# Define robot and scene parameters
robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "ramp" #"flat" #"random_boxes" "ramp" #"perlin" #"stairs" #"flat" #"random_boxes"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = config.mpc_frequency
state_observables_names = tuple(RewQuadrupedEnv.ALL_OBS)  # return all available state observables
# Initialize simulation environment
sim_frequency = 200.0
env = RewQuadrupedEnv(robot=robot_name,
                   scene=scene_name,
                   sim_dt = 1/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=0.7,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )


env = OurGymnasiumWrapper(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True)
models["value"] = Critic(env.observation_space, None, device)
# models["critic_2"] = Critic(env.observation_space, env.action_space, device)
# models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
# models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
TRPO_QUADRUPED_CONFIG = {
    "rollouts": 256,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 4,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "value_learning_rate": 2e-5,            # value learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 1000,           # learning starts after this many steps

    "grad_norm_clip": 1.,          # clipping coefficient for the norm of the gradients
    "value_loss_scale": 1.0,        # value loss scaling factor

    "damping": 0.1,                     # damping coefficient for computing the Hessian-vector product
    "max_kl_divergence": 0.01,          # maximum KL divergence between old and new policy
    "conjugate_gradient_steps": 20,     # maximum number of iterations for the conjugate gradient algorithm
    "max_backtrack_steps": 10,          # maximum number of backtracking steps during line search
    "accept_ratio": 0.5,                # accept ratio for the line search loss improvement
    "step_fraction": 1.0,               # fraction of the step size for the line search

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
cfg = TRPO_QUADRUPED_CONFIG.copy()
cfg["discount_factor"] = 0.99
cfg["batch_size"] = 64
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 75
cfg["experiment"]["checkpoint_interval"] = 100000
cfg["experiment"]["directory"] = "runs/torch/trpo/QuadrupedEnv"

agent = TRPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
