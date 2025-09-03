import os
import sys
import time
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
import jax.numpy as jnp
import jax
import mujoco
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from gym_quadruped.quadruped_env import QuadrupedEnv
import copy
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
import utils.mpc_wrapper as mpc_wrapper
import config.config_aliengo as config
from timeit import default_timer as timer
import pdb
sys.path.append("algorithm")
import algorithm.fcn as fcn
import algorithm.memory as memory
from algorithm.init import StochasticDecisionTransformer
from transformers import DecisionTransformerConfig, DecisionTransformerModel
from sophia import SophiaG

def get_model_input_observation(obs, dtype, device):
    return torch.tensor(np.concatenate([
                                        obs['base_pos'],
                                        obs['base_ori_quat_wxyz'],
                                        obs['qpos_js'],
                                        obs['base_lin_vel'],
                                        obs['base_ang_vel'],
                                        obs['qvel_js'],
                                        obs['feet_pos'],
                                        obs['contact_forces']]),dtype=dtype, device=device)

def compute_returns_to_go_batch(reward_batch: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """
    Compute discounted returns-to-go for a batch of reward sequences.

    Args:
        reward_batch: [B, T] tensor of rewards
        gamma: discount factor

    Returns:
        returns: [B, T] tensor of returns-to-go
    """
    B, T = reward_batch.shape
    returns = torch.zeros_like(reward_batch)
    running_return = torch.zeros(B, dtype=reward_batch.dtype, device=reward_batch.device)
    for t in reversed(range(T)):
        running_return = reward_batch[:, t] + gamma * running_return
        returns[:, t] = running_return
    return returns

class RewQuadrupedEnv(QuadrupedEnv):
    def _compute_reward(self):
        lin_rew = np.sum(np.exp(-self.base_lin_vel_err()))
        ang_rew = np.sum(np.exp(-self.base_ang_vel_err()))
        torque_penalty = 0.1 * (np.sum(self.mjData.ctrl >= config.max_torque) + np.sum(self.mjData.ctrl <= config.min_torque))
        work_penalty = 0.01 * self.work
        return lin_rew+ang_rew-torque_penalty-work_penalty

# class StochasticDecisionTransformer(DecisionTransformerModel):
#     def __init__(self, config):
#         super().__init__(config)
#         act_dim = config.act_dim
        
#         self.mean_head = nn.Linear(config.hidden_size, act_dim)
#         self.logstd_head = nn.Linear(config.hidden_size, act_dim)
        
#     def forward(self, *args, **kwargs):
#         kwargs["output_hidden_states"] = True
#         kwargs["return_dict"] = True
#         out = super().forward(*args, **kwargs)
        
#         batch_size = out.last_hidden_state.shape[0]
#         trajectory_len = out.last_hidden_state.shape[1] // 3
        
#         last_hidden_action = out.last_hidden_state.reshape(batch_size, trajectory_len, 3, self.hidden_size).permute(0, 2, 1, 3)[:, 1]
        
#         mean = self.mean_head(last_hidden_action)
#         logstd = torch.clamp(self.logstd_head(last_hidden_action), min=-20, max=2)
        
#         return out.state_preds, (mean, logstd), out.return_preds

class MPCGuidanceAgent:
    def __init__(self, config_dict=None):
        """Initialize MPC Guidance Agent
        
        Args:
            config_dict: Configuration dictionary with training parameters
        """
        # Default configuration
        self.cfg = {
            "timesteps": 1000000,
            "batch_size": 32,
            "learning_rate": 1.0e-5,
            "learning_starts":1000,
            "train_freq": 200,
            "eval_freq": 2000,
            "num_minibatch_updates": 16,
            "trajectory_len": 25,
            "epochs": 1,
            # "log_frequency": 100,
            "state_dim": 61,
            "act_dim": 12,
            "alpha": 0.1,
            "target_entropy": 0.2,
            "gamma": 0.99,
            "grad_norm" : 1.0,
            "memory_capacity": 10000,
            "hidden_size": 128,
            "max_ep_len": 4096,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dtype": torch.float32,
            "result_dir": "/home/satya/Robot/result/"
        }
        
        if config_dict:
            self.cfg.update(config_dict)
            
        self.device = torch.device(self.cfg["device"])
        self.dtype = self.cfg["dtype"]
        
        # Initialize memory
        self.replay_memory = memory.ReplayMemory(self.cfg["memory_capacity"])
        
        # Initialize model
        self._init_model()
        
        # Initialize optimizer
        self.optimizer = SophiaG(self.model.parameters(), lr=self.cfg["learning_rate"])
        
        # Training tracking
        self.running_loss = torch.tensor([], dtype=self.dtype, device=self.device)
        self.running_entropy = torch.tensor([], dtype=self.dtype, device=self.device)
        
        # Environment and MPC will be set externally
        self.env = None
        self.mpc = None
        
    def _init_model(self):
        """Initialize the transformer model"""
        configuration = DecisionTransformerConfig(
            state_dim=self.cfg["state_dim"],
            act_dim=self.cfg["act_dim"],
            hidden_size=self.cfg["hidden_size"],
            max_ep_len=self.cfg["max_ep_len"],
            action_tanh=True
        )
        self.model = StochasticDecisionTransformer(configuration).to(self.dtype).to(self.device)
        
    def set_environment(self, env):
        """Set the environment"""
        self.env = env
        
    def set_mpc_controller(self, mpc):
        """Set the MPC controller"""
        self.mpc = mpc
        
    def act(self, states, timesteps, returns_to_go, actions=None):
        """Get actions from the transformer model
        
        Args:
            states: Current states
            timesteps: Timestep information
            returns_to_go: Returns to go
            actions: Previous actions (for conditioning)
            
        Returns:
            actions: Predicted actions
        """
        self.model.eval()
        # print(self.model)
        with torch.no_grad():
            if actions is None:
                # For inference, we might need to handle this differently
                # This is a simplified version
                actions = torch.zeros(states.shape[0], states.shape[1], self.cfg["act_dim"], 
                                    device=self.device, dtype=self.dtype)
            
            rewards = torch.zeros(states.shape[0], states.shape[1], 1, 
                                device=self.device, dtype=self.dtype)
            attention_mask = torch.ones(states.shape[0], states.shape[1], device=self.device)
            # print(timesteps.shape)
            _, (action_pred_mean, action_pred_logstd), _ = self.model.evaluate(timesteps.shape[1],
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask
            )
            
            # Sample from the distribution
            action_pred_dist = torch.distributions.normal.Normal(action_pred_mean, torch.exp(action_pred_logstd))
            actions = action_pred_dist.sample()
            
        return actions
        
    def record_transition(self, trajectory_seq):
        """Record a trajectory sequence in memory
        
        Args:
            trajectory_seq: Tuple of (rewards, states, actions)
        """
        self.replay_memory.push(trajectory_seq)
        
    def update(self):
        """Update the model using recorded trajectories"""
        if len(self.replay_memory) < self.cfg["batch_size"]:
            return
            
        self.model.train()
        
        for epoch in range(self.cfg["epochs"]):
            for update_step in range(len(self.replay_memory) // self.cfg["batch_size"]):
                if update_step < self.cfg["num_minibatch_updates"]:
                    # Sample batch from memory
                    trajectory_batch = self.replay_memory.sample(self.cfg["batch_size"])
                    
                    # Unpack trajectories
                    reward_seq_batch, state_seq_batch, action_seq_batch = zip(*trajectory_batch)
                    
                    # Convert to tensors
                    rewards_tensor = torch.stack([torch.from_numpy(np.array(r)) for r in reward_seq_batch]).to(device=self.device, dtype=self.dtype)
                    states_tensor = torch.stack([torch.from_numpy(np.array(s)) for s in state_seq_batch]).to(device=self.device, dtype=self.dtype)
                    actions_tensor = torch.stack([torch.from_numpy(np.array(a)) for a in action_seq_batch]).to(device=self.device, dtype=self.dtype)
                    
                    # Compute returns to go
                    returns_tensor = compute_returns_to_go_batch(rewards_tensor, gamma=self.cfg["gamma"])
                    
                    # Prepare tensors for model
                    rewards_tensor = rewards_tensor.unsqueeze(-1)
                    returns_tensor = returns_tensor.unsqueeze(-1)
                    
                    timesteps = torch.arange(self.cfg["trajectory_len"]).repeat(self.cfg["batch_size"])
                    timesteps_tensor = timesteps.detach().clone().to(dtype=torch.long, device=self.device).reshape(self.cfg["batch_size"], self.cfg["trajectory_len"])
                    # pdb.set_trace()
                    # Forward pass
                    _, (action_pred_mean, action_pred_logstd), _ = self.model(
                        states=states_tensor,
                        actions=actions_tensor,
                        rewards=rewards_tensor,
                        returns_to_go=returns_tensor,
                        timesteps=timesteps_tensor,
                        attention_mask=torch.ones(self.cfg["batch_size"], self.cfg["trajectory_len"], device=self.device)
                    )
                    
                    # Compute loss
                    # pdb.set_trace()
                    # print("NaN in loc:", torch.isnan(action_pred_mean).any())
                    # print("Inf in loc:", torch.isinf(action_pred_mean).any())
                    # print("NaN in scale:", torch.isnan(action_pred_logstd).any())
                    # print("Negative in scale:", (action_pred_logstd <= 0).any())
                    # print("Inf in scale:", torch.isinf(action_pred_logstd).any())
                    # print(torch.norm(action_pred_mean), torch.norm(action_pred_logstd))
                    # action_pred_mean = torch.clamp(action_pred_mean, min=-1, max=1)
                    # action_pred_logstd = torch.clamp(action_pred_logstd, min=-1, max=1)
                    # action_pred_logstd = torch.nn.functional.softplus(action_pred_logstd) + 1e-6
                    action_pred_dist = torch.distributions.normal.Normal(action_pred_mean, torch.exp(action_pred_logstd))
                    action_pred_entropy = action_pred_dist.entropy().sum(dim=-1)
                    nllloss = -action_pred_dist.log_prob(actions_tensor)
                    mean_entropy = action_pred_entropy.mean()
                    loss = nllloss.sum(dim=-1).mean() + self.cfg["alpha"] * (self.cfg["target_entropy"] - mean_entropy)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_norm"], norm_type=2.0)
                    self.optimizer.step()
                    
                    # Track loss & entropy
                    self.running_loss = torch.cat([self.running_loss, loss.detach().unsqueeze(0)])
                    self.running_entropy = torch.cat([self.running_entropy, mean_entropy.detach().unsqueeze(0)])
                
            # print(f"Epoch {epoch} | Loss: {self.running_loss[-1].item():.4f}")
            
    def save_model(self, path):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.running_loss,
            'entropy': self.running_entropy,
        }, path)
        
    def load_model(self, path):
        """Load the model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.running_loss = checkpoint['loss']
        
    def save_loss(self):
        """Save loss history"""
        torch.save(self.running_loss, self.cfg["result_dir"] + 'loss_quad_mpc_guidance.pth')
        torch.save(self.running_entropy, self.cfg["result_dir"] + 'ent_quad_mpc_guidance.pth')

def main():
    """Main training loop"""
    
    # Define robot and scene parameters
    robot_name = "aliengo"
    scene_name = "ramp"
    robot_feet_geom_names = dict(FR='FR', FL='FL', RR='RR', RL='RL')
    robot_leg_joints = dict(
        FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'],
        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'],
        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'],
        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
    )
    mpc_frequency = config.mpc_frequency
    state_observables_names = tuple(RewQuadrupedEnv.ALL_OBS)
    
    # Initialize simulation environment
    sim_frequency = 200.0
    env = RewQuadrupedEnv(
        robot=robot_name, # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
        scene=scene_name, #"flat" #"random_boxes" "ramp" #"perlin" #"stairs" #"flat" #"random_boxes"
        sim_dt=1/sim_frequency, # Simulation time step [s]
        ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
        ground_friction_coeff=0.7, # pass a float for a fixed value
        base_vel_command_type="human", # "forward", "random", "forward+rotate", "human"
        state_obs_names=state_observables_names, # Desired quantities in the 'state'
    )
    state = env.reset(random=False)
    
    # Initialize MPC controller
    mpc = mpc_wrapper.MPCControllerWrapper(config)
    env.mjData.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    
    # Initialize agent
    agent = MPCGuidanceAgent()
    agent.set_environment(env)
    agent.set_mpc_controller(mpc)
    
    # Initialize simulation variables
    counter = 0
    tau = jnp.zeros(config.n_joints)
    delay = 1
    q = config.q0.copy()
    dq = jnp.zeros(config.n_joints)
    mpc.robot_height = config.robot_height
    mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())
    
    print("Running MPC Guidance Training...")
    # Reset the robot after max episode length and start training again
    # if counter % agent.cfg["max_ep_len"] == 0:
    state = env.reset(random=False)
    #initialize inputs to transformer
    X_start = get_model_input_observation(obs=state, dtype=agent.cfg["dtype"], device=agent.cfg["device"])
    obs = X_start.unsqueeze(0).unsqueeze(0)
    t_step = torch.tensor([1], dtype=torch.long, device=agent.cfg["device"]).unsqueeze(0)
    rtg = torch.tensor([100], dtype=agent.cfg["dtype"], device=agent.cfg["device"]).unsqueeze(0).unsqueeze(0)
    act = torch.zeros(obs.shape[0], obs.shape[1], agent.cfg["act_dim"], device=agent.cfg["device"], dtype=agent.cfg["dtype"])

    # Main simulation loop
    while counter < agent.cfg["timesteps"]: # Break command added here for clarity
        
        # if counter == 0 or counter < agent.cfg["max_episode_length"]:
        #     # Create new state_tensor, timestep_tensor, return_to_go_tensor
            

        qpos = env.mjData.qpos.copy()
        qvel = env.mjData.qvel.copy()
        
        # if (counter % (sim_frequency / mpc_frequency) == 0 or counter == 0):
            
        # Get reference velocities
        ref_base_lin_vel = env._ref_base_lin_vel_H
        ref_base_ang_vel = np.array([0., 0., env._ref_base_ang_yaw_dot])
        
        input_cmd = np.array([
            ref_base_lin_vel[0], ref_base_lin_vel[1], ref_base_lin_vel[2],
            ref_base_ang_vel[0], ref_base_ang_vel[1], ref_base_ang_vel[2],
            config.robot_height
        ])
        
        contact_temp, _ = env.feet_contact_state()
        contact = np.array([contact_temp[robot_feet_geom_names[leg]] for leg in ['FL', 'FR', 'RL', 'RR']])
        
        # if (counter % delay == 0) and (counter != 0):
            # for i in range(delay):
        # qpos = env.mjData.qpos.copy()
        # qvel = env.mjData.qvel.copy()
        tau, q, dq, X, U, stage_cost = mpc.run(qpos, qvel, input_cmd, contact)
        tau_fb = 10*(q-qpos[7:7+config.n_joints]) - 2*(qvel[6:6+config.n_joints])
        # state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb)
            # counter += 1
        
        # Run MPC
        # start = timer()
        # tau, q, dq, X, U, stage_cost = mpc.run(qpos, qvel, input_cmd, contact)
        
        # Record trajectory for learning
        trajectory_seq = (-stage_cost[:-1], X[:-1], U)
        agent.record_transition(trajectory_seq)

        # Truncate model inputs for forward pass
        obs_seq = obs[:, -1:, :]
        t_step_seq = t_step[:, -1:]
        rtg_seq = rtg[:, -1:, :]
        act_seq = act[:, -1:, :]
        # pdb.set_trace()
        U_actual = agent.act(states=obs_seq, timesteps=t_step_seq, returns_to_go=rtg_seq, actions=act_seq)
        # print(torch.norm(obs_seq), torch.norm(rtg_seq),torch.norm(act_seq), torch.norm(U_actual))
        
        # Apply control
        # tau_fb = 10*(q-qpos[7:7+config.n_joints]) - 2*(qvel[6:6+config.n_joints])
        if counter < 2000:
            state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb) #U_actual
        else:
            state, reward, is_terminated, is_truncated, info = env.step(action=U_actual.cpu().numpy())
        
        counter += 1
        X_new = get_model_input_observation(obs=state, dtype=agent.cfg["dtype"], device=agent.cfg["device"]).unsqueeze(0).unsqueeze(0)
        obs = torch.cat([obs, X_new], dim=1)
        t_step = torch.cat([t_step, torch.tensor([[t_step_seq.size(1)+1]], dtype=torch.long, device=agent.cfg["device"])], dim=1)
        rtg = torch.cat([rtg, (rtg[:, -1] - reward).unsqueeze(0)], dim=1)
        act = torch.cat([act, U_actual], dim=1)
        # pdb.set_trace()
        # Training
        if counter >= agent.cfg["learning_starts"] and len(agent.replay_memory) > agent.cfg["batch_size"]:
            # print(counter)
            if counter % agent.cfg["train_freq"] = 0:
                agent.update()
                agent.save_loss()
                print(f"Step {counter} | Memory size: {len(agent.replay_memory)} | Latest loss: {agent.running_loss[-1].item():.4f} | Latest Entropy: {agent.running_entropy[-1].item():.4f}")
            
            # if counter % agent.cfg["log_frequency"] == 0:
            #     agent.save_loss()
            #     print(f"Step {counter} | Memory size: {len(agent.replay_memory)} | Latest loss: {agent.running_loss[-1].item():.4f}")
        
        # # Optional: Add termination condition
        # if counter > agent.cfg["timesteps"]:  # Example termination condition
        #     break

    
    # Save final model
    agent.save_model(agent.cfg["result_dir"] + "mpc_guidance_model.pth")
    print("Training completed!")

if __name__ == "__main__":
    main()
