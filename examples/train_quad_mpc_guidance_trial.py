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

def sanity_check_output_tensor(tensor, dim):
    mask = ~torch.isnan(tensor).any(dim=dim)   # [32, 25] → True if row has no NaN
    tensor[mask] = 0.0
    # tensor = tensor[mask].view(tensor.size(0), -1, tensor.size(2))
    return tensor

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
        return 0.1 * ( lin_rew+ang_rew - torque_penalty - 0.1 * work_penalty)

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
            "learning_rate": 8.0e-5,
            "learning_starts":1000,
            "train_freq": 10,
            "eval_freq": 1000,
            "num_minibatch_updates": 16,
            "trajectory_len": 25,
            "epochs": 1,
            "log_frequency": 100,
            "state_dim": 61,
            "act_dim": 12,
            "alpha": 2e-3,
            "target_entropy": 0.1, # 0.1
            "gamma": 0.995,
            "grad_norm" : 1.0,
            "memory_capacity": 100000,
            "hidden_size": 256,
            "max_ep_len": 128, #4096
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
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["learning_rate"])
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
                    # print("action_pred_mean", action_pred_mean[:, -1, :])
                    # print(" action_pred_logstd", action_pred_logstd[:, -1, :])
                    # action_pred_mean = sanity_check_output_tensor(action_pred_mean, dim=1)
                    # action_pred_logstd = sanity_check_output_tensor(action_pred_logstd, dim=1)
                    action_pred_dist = torch.distributions.normal.Normal(action_pred_mean, torch.exp(action_pred_logstd))
                    action_pred_entropy = action_pred_dist.entropy().sum(dim=-1)
                    # print("entropy", action_pred_entropy.mean())
                    nllloss = -action_pred_dist.log_prob(actions_tensor)
                    mean_entropy = action_pred_entropy.mean()
                    loss = nllloss.sum(dim=-1).mean() + self.cfg["alpha"] * (self.cfg["target_entropy"] - action_pred_entropy.mean())
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_norm"], norm_type=2.0)
                    self.optimizer.step(bs=self.cfg["batch_size"]*self.cfg["num_minibatch_updates"])
                    
                    # Track loss & entropy
                    self.running_loss = torch.cat([self.running_loss, loss.detach().unsqueeze(0)])
                    self.running_entropy = torch.cat([self.running_entropy, mean_entropy.detach().unsqueeze(0)])
                
            # print(f"Epoch {epoch} | Loss: {self.running_loss[-1].item():.4f}")
            
    def save_model(self, path):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.running_loss
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

    def evaluate_agent(self, render=False, num_episodes=1):
        from tqdm import tqdm

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
        sim_frequency = 200
        eval_env = RewQuadrupedEnv(
            robot=robot_name, # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
            scene=scene_name, #"flat" #"random_boxes" "ramp" #"perlin" #"stairs" #"flat" #"random_boxes"
            sim_dt=1/sim_frequency, # Simulation time step [s]
            ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
            ground_friction_coeff=0.7, # pass a float for a fixed value
            base_vel_command_type="human", # "forward", "random", "forward+rotate", "human"
            state_obs_names=state_observables_names, # Desired quantities in the 'state'
        )
        obs = eval_env.reset(random=False)
        if render:
            eval_env.render(tint_robot=True)
        ep_rew_array = []
        for ep in range(num_episodes):
            obs = eval_env.reset()
            X_start = get_model_input_observation(obs=obs, dtype=self.cfg["dtype"], device=self.cfg["device"])
            obs = X_start.unsqueeze(0).unsqueeze(0)
            t_step = torch.tensor([1], dtype=torch.long, device=self.cfg["device"]).unsqueeze(0)
            rtg = torch.tensor([5000], dtype=self.cfg["dtype"], device=self.cfg["device"]).unsqueeze(0).unsqueeze(0)
            act = torch.zeros(obs.shape[0], obs.shape[1], self.cfg["act_dim"], device=self.cfg["device"], dtype=self.cfg["dtype"])
            obs_seq = []
            t_step_seq = []
            rtg_seq = []
            act_seq = []
            total_rew = 0
            for step in tqdm(range(200), desc=f'Episode {ep}', leave=False):
                qpos, qvel = eval_env.mjData.qpos, eval_env.mjData.qvel

                obs_seq = obs[:, -self.cfg["max_ep_len"]:-1 if t_step.shape[-1] > 1 else None, :]
                t_step_seq = t_step[:, -self.cfg["max_ep_len"]:-1 if t_step.shape[-1] > 1 else None]
                rtg_seq = rtg[:, -self.cfg["max_ep_len"]:-1 if t_step.shape[-1] > 1 else None, :]
                act_seq = act[:, -self.cfg["max_ep_len"]:-1 if t_step.shape[-1] > 1 else None, :]

                action = self.act(states=obs_seq, timesteps=t_step_seq, returns_to_go=rtg_seq, actions=act_seq)
                state, reward, is_terminated, is_truncated, info = eval_env.step(action=action.detach().cpu().numpy())
                total_rew += reward
                # print(f"Kinetic energy: {state['kinetic_energy'].item():.3e} \t Work done: {state['work'].item():.3e}")
                for state_obs_name in state_observables_names:
                    assert state_obs_name in state, f'Missing state observation: {state_obs_name}'
                    assert state[state_obs_name] is not None, f'Invalid state observation: {state_obs_name}'

                if is_terminated:
                    pass
                    # Handle terminal states here. Terminal states are contacts with ground with any geom but feet.

                # The environment enables also to visualize ghost robot configurations for debugging purposes.
                # These ghost/decorative robots are not simulated, rather only displayed in the viewer.
                # These robot's config are given by a qpos array.
                qpos_ghost1, qpos_ghost2 = np.array(qpos), np.array(qpos)
                qpos_ghost1[0] += 1.0
                qpos_ghost2[0] -= 1.0
                if render:
                    eval_env.render(ghost_qpos=(qpos_ghost1, qpos_ghost2), ghost_alpha=(0.1, 0.5))
            ep_rew_array.append(total_rew)
        eval_env.close()
        ep_rew_array = np.array(ep_rew_array)
        print(f"Mean Evaluation Total Rewards: {ep_rew_array.mean():.4f}")

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
    print(agent.model)
    # Initialize simulation variables
    counter = 0
    tau = jnp.zeros(config.n_joints)
    tau_old = jnp.zeros(config.n_joints)
    delay = int(0.007*sim_frequency)
    q = config.q0.copy()
    dq = jnp.zeros(config.n_joints)
    mpc.robot_height = config.robot_height
    mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())
    
    print("Running MPC Guidance Training...")
    # Reset the robot after max episode length and start training again
    # if counter % agent.cfg["max_ep_len"] == 0:
    #     state = env.reset(random=False)
    #     #initialize inputs to transformer
    #     X_start = get_model_input_observation(obs=state, dtype=agent.cfg["dtype"], device=agent.cfg["device"])
    #     obs = X_start.unsqueeze(0).unsqueeze(0)
    #     t_step = torch.tensor([1], dtype=torch.long, device=agent.cfg["device"]).unsqueeze(0)
    #     rtg = torch.tensor([100], dtype=agent.cfg["dtype"], device=agent.cfg["device"]).unsqueeze(0).unsqueeze(0)
    #     act = torch.zeros(obs.shape[0], obs.shape[1], agent.cfg["act_dim"], device=agent.cfg["device"], dtype=agent.cfg["dtype"])

    # Main simulation loop
    while counter < agent.cfg["timesteps"]: # Break command added here for clarity

        if counter % agent.cfg["max_ep_len"] == 0:
            state = env.reset(random=False)
            #initialize inputs to transformer
            X_start = get_model_input_observation(obs=state, dtype=agent.cfg["dtype"], device=agent.cfg["device"])
            obs = X_start.unsqueeze(0).unsqueeze(0)
            t_step = torch.tensor([1], dtype=torch.long, device=agent.cfg["device"]).unsqueeze(0)
            rtg = torch.tensor([5000], dtype=agent.cfg["dtype"], device=agent.cfg["device"]).unsqueeze(0).unsqueeze(0)
            act = torch.zeros(obs.shape[0], obs.shape[1], agent.cfg["act_dim"], device=agent.cfg["device"], dtype=agent.cfg["dtype"])
            obs_seq = []
            t_step_seq = []
            rtg_seq = []
            act_seq = []
        
        # if counter == 0 or counter < agent.cfg["max_episode_length"]:
        #     # Create new state_tensor, timestep_tensor, return_to_go_tensor
            

        qpos = env.mjData.qpos.copy()
        qvel = env.mjData.qvel.copy()
        
        if (counter % (sim_frequency / mpc_frequency) == 0 or counter == 0):
            
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
            
            if (counter != 0):
                for i in range(delay):
                    # print("delay", delay)
                    qpos = env.mjData.qpos.copy()
                    qvel = env.mjData.qvel.copy()
                    tau, q, dq, X, U, stage_cost = mpc.run(qpos, qvel, input_cmd, contact)
                    tau_fb = 10*(q-qpos[7:7+config.n_joints]) - 2*(qvel[6:6+config.n_joints])
                    state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb)
                    counter += 1
        
            # Run MPC
            # start = timer()
            # qpos = env.mjData.qpos.copy()
            # qvel = env.mjData.qvel.copy()
            tau, q, dq, X, U, stage_cost = mpc.run(qpos, qvel, input_cmd, contact)
        
            # Record trajectory for learning
            trajectory_seq = (-stage_cost[:-1], X[:-1], U)
            agent.record_transition(trajectory_seq)

        # Truncate model inputs for forward pass
        # print(obs.shape, t_step.shape)
        obs_seq = obs[:, -agent.cfg["max_ep_len"]:-1 if t_step.shape[-1] > 1 else None, :]
        t_step_seq = t_step[:, -agent.cfg["max_ep_len"]:-1 if t_step.shape[-1] > 1 else None]
        rtg_seq = rtg[:, -agent.cfg["max_ep_len"]:-1 if t_step.shape[-1] > 1 else None, :]
        act_seq = act[:, -agent.cfg["max_ep_len"]:-1 if t_step.shape[-1] > 1 else None, :]
        # print(obs_seq.shape, t_step_seq.shape, rtg_seq.shape, act_seq.shape)
        # pdb.set_trace()
        # print(counter)
        U_actual = agent.act(states=obs_seq, timesteps=t_step_seq, returns_to_go=rtg_seq, actions=act_seq)
        # print(torch.norm(obs_seq), torch.norm(rtg_seq),torch.norm(act_seq), torch.norm(U_actual))
        
        # Apply control
        tau_fb = 10*(q-qpos[7:7+config.n_joints]) - 2*(qvel[6:6+config.n_joints])
        # if counter < 2000:
        state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb) #U_actual
        # if counter%10 == 0:
            # print(counter, tau + tau_fb)
        # else:
        #     state, reward, is_terminated, is_truncated, info = env.step(action=U_actual.detach().cpu().numpy())
        
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
            if counter % agent.cfg["train_freq"] == 0:
                agent.update()

            if counter % agent.cfg["log_frequency"] == 0:
                agent.save_loss()
                print(f"Train Step {counter} | Memory size: {len(agent.replay_memory)} | Latest loss: {agent.running_loss[-1].item():.4f} | Latest Entropy: {agent.running_entropy[-1].item():.4f}")

            if counter % agent.cfg["eval_freq"] == 0:
                agent.evaluate_agent(num_episodes=10)
        
        # # Optional: Add termination condition
        # if counter > agent.cfg["timesteps"]:  # Example termination condition
        #     break

    
    # Save final model
    agent.save_model(agent.cfg["result_dir"] + "mpc_guidance_model.pth")
    print("Training completed!")

if __name__ == "__main__":
    main()
