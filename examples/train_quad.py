import os
import sys
import time
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
import jax.numpy as jnp
import jax
import mujoco

# Update JAX configuration
# jax.config.update("jax_compilation_cache_dir", "./jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
 
import numpy as np
import primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
from functools import partial
from gym_quadruped.quadruped_env import QuadrupedEnv
import copy
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
 
import utils.mpc_wrapper as mpc_wrapper
import config.config_aliengo as config

from timeit import default_timer as timer
# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)


sys.path.append("algorithm")
from algorithm.init import *
import algorithm.fcn as fcn
import algorithm.memory as memory



 
# Define robot and scene parameters
robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat" #"random_boxes"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = config.mpc_frequency
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables
 
# Initialize simulation environment
sim_frequency = 200.0
env = QuadrupedEnv(robot=robot_name,
                   scene=scene_name,
                   sim_dt = 1/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=0.7,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )
obs = env.reset(random=False)
# Define the MPC wrapper
mpc = mpc_wrapper.MPCControllerWrapper(config)
env.mjData.qpos = jnp.concatenate([config.p0, config.quat0,config.q0])
# env.render()
ids = []
# for i in range(8):
#      ids.append(render_vector(env.viewer,
#               np.zeros(3),
#               np.zeros(3),
#               0.1,
#               np.array([1, 0, 0, 1])))
counter = 0
# Main simulation loop
tau = jnp.zeros(config.n_joints)
tau_old = jnp.zeros(config.n_joints)
delay = int(0.007*sim_frequency)
# print('Delay: ',delay)

q = config.q0.copy()
dq = jnp.zeros(config.n_joints)
mpc_time = 0
mpc.robot_height = config.robot_height
mpc.reset(env.mjData.qpos.copy(),env.mjData.qvel.copy())
# while env.viewer.is_running():
print("Running...")
while True:

    qpos = env.mjData.qpos.copy()
    qvel = env.mjData.qvel.copy()
    if (counter % (sim_frequency / mpc_frequency) == 0 or counter == 0):
    
 
        ref_base_lin_vel = env._ref_base_lin_vel_H
        ref_base_ang_vel =  np.array([0., 0., env._ref_base_ang_yaw_dot])
 
        
        input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           config.robot_height])
        
        contact_temp, _ = env.feet_contact_state()
        
        contact = np.array([contact_temp[robot_feet_geom_names[leg]] for leg in ['FL','FR','RL','RR']])

        if counter != 0:
            for i in range(delay):
                qpos = env.mjData.qpos.copy()
                qvel = env.mjData.qvel.copy()
                # tau_fb = K@(x-np.concatenate([qpos,qvel]))

                tau_fb = 10*(q-qpos[7:7+config.n_joints]) -2*(qvel[6:6+config.n_joints])

                ## rolling out: #########################################################
                action_vec = tau + tau_fb
                state_vec = fcn.flatten_state(state)
                transition = (reward, state_vec, action_vec)
                
                if len(trajectory_seq) < TRAJECTORY_LEN:
                    trajectory_seq.append(transition)
                else:
                    replay_memory.push(trajectory_seq)
                    trajectory_seq =  []
                #########################################################################

                state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb)
                counter += 1
        start = timer()
        tau, q, dq = mpc.run(qpos,qvel,input,contact)   
        stop = timer()
        # print("Time taken for MPC: ", stop-start)   

        stop = timer()
        # for i in range(4):
        #     render_sphere(env.viewer,
        #                   collision_point[3*i:3*i+3],
        #                   0.2,
        #                   np.array([1, 0, 0, 0.5]),
        #                   ids[i])

    tau_fb = 10*(q-qpos[7:7+config.n_joints])-2*(qvel[6:6+config.n_joints])
    state, reward, is_terminated, is_truncated, info = env.step(action= tau + tau_fb)

    # time.sleep(0.1)
    counter += 1
    # env.render()


    ## training: ############################################
    if counter % 10 == 0 and len(replay_memory) > BATCH_SIZE:
        model.train()
        optimizer.zero_grad()

        for epoch in range(EPOCHS):
            for _ in range(len(replay_memory) // BATCH_SIZE):

                reward_seq_batch = []
                state_seq_batch = []
                action_seq_batch = []
                trajectory_batch = replay_memory.sample(BATCH_SIZE)  # a list (batch) of list (trajectory) of tuples (transitions)
                # for traj in trajectory_batch:
                #     for trans in traj:
                #         reward_seq_batch.append(trans[0])  
                #         state_seq_batch.append(trans[1])   
                #         action_seq_batch.append(trans[2])

                reward_seq_batch, state_seq_batch, action_seq_batch = zip(*[
                    trans for traj in trajectory_batch for trans in traj
                ])

                timesteps = torch.arange(TRAJECTORY_LEN).repeat(BATCH_SIZE)
                return_seq_batch = reward_seq_batch #!!!!!!!!

                states_tensor = torch.tensor(state_seq_batch, dtype=dtype, device=device).reshape(BATCH_SIZE, TRAJECTORY_LEN, STATE_DIM)
                actions_tensor = torch.tensor(action_seq_batch, dtype=dtype, device=device).reshape(BATCH_SIZE, TRAJECTORY_LEN, ACT_DIM)
                rewards_tensor = torch.tensor(reward_seq_batch, dtype=dtype, device=device).reshape(BATCH_SIZE, TRAJECTORY_LEN, 1)
                returns_tensor = torch.tensor(return_seq_batch, dtype=dtype, device=device).reshape(BATCH_SIZE, TRAJECTORY_LEN, 1)
                timesteps_tensor = torch.tensor(timesteps, dtype=torch.long, device=device).reshape(BATCH_SIZE, TRAJECTORY_LEN)

                _, action_preds, _ = model(
                    states=states_tensor,
                    actions=actions_tensor,
                    rewards=rewards_tensor,
                    returns_to_go=returns_tensor,
                    timesteps=timesteps_tensor,
                    attention_mask=torch.ones(BATCH_SIZE, TRAJECTORY_LEN, device=device),
                    return_dict=False
                )

                loss = loss_fcn(action_preds,actions_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                running_loss = torch.cat([running_loss, loss.detach().unsqueeze(0)])
                if counter % LOG_FREQUENCY == 0:
                    torch.save(running_loss, result_dir + 'loss_quad_adam.pth')
                print(f"Epoch {epoch} | Loss: {running_loss[-1].item():.4f}")

            print("-------------------------")


        # print("TEST 1 = ", len(trajectory_batch))            # len of batch
        # print("TEST 2 = ", len(trajectory_batch[0]))         # len of trajectory
        # print("TEST 3 = ", trajectory_batch[0])            # list (trajectory) of tuples (transitions)
        # print("TEST 4 = ", trajectory_batch[0][0])           # a tuple (transition)
        # print("TEST 5 = ", trajectory_batch[0][0][0])        # reward
        # print("TEST 6 = ", trajectory_batch[0][0][1])        # state
        # print("TEST 7 = ", trajectory_batch[0][0][2])        # action
        # print(tau.to(device)) #Error!
    #########################################################