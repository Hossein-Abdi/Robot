import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
import jax
# jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
# jax.config.update("jax_compilation_cache_dir", "./jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import numpy as np
import mujoco
import mujoco.viewer
import numpy as np
from gym_quadruped.utils.mujoco.visual import render_sphere ,render_vector
import utils.mpc_wrapper as mpc_wrapper
import config.config_talos as config

sys.path.append("algorithm")
from algorithm.init import *
import algorithm.fcn as fcn
import algorithm.memory as memory



env = mujoco.MjModel.from_xml_path(dir_path+'/../data/pal_talos/talos_motor_rough.xml')
data = mujoco.MjData(env)
sim_frequency = 500.0
env.opt.timestep = 1/sim_frequency

contact_id = []
for name in config.contact_frame:
    contact_id.append(mujoco.mj_name2id(env,mujoco.mjtObj.mjOBJ_GEOM,name))
mpc = mpc_wrapper.MPCControllerWrapper(config)
data.qpos = jnp.concatenate([config.p0, config.quat0,config.q0])

from timeit import default_timer as timer

ids = []
tau = jnp.zeros(config.n_joints)

mujoco.mj_step(env, data)
# viewer.sync()
delay = int(0.005*sim_frequency)
# print('Delay: ',delay)
mpc.robot_height = config.robot_height
mpc.reset(data.qpos.copy(),data.qvel.copy())
counter = 0
# while viewer.is_running():
print("Running...")
while True:
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    if counter % (sim_frequency / config.mpc_frequency) == 0 or counter == 0:  
        if counter != 0:
            for i in range(delay):
                qpos = data.qpos.copy()
                qvel = data.qvel.copy()
                tau_fb = -3*(qvel[6:6+config.n_joints])
                data.ctrl = tau + tau_fb
                ## rolling out: #########################################################
                action_vec = data.ctrl.copy()
                state_vec = np.concatenate([qpos, qvel], axis=-1)
                # state_vec = jnp.array(state)
                # state_vec = fcn.flatten_state(state)
                reward = np.array([0]) # !!!!!!!!
                transition = (reward, state_vec, action_vec)
                
                if len(trajectory_seq) < TRAJECTORY_LEN:
                    trajectory_seq.append(transition)
                else:
                    replay_memory.push(trajectory_seq)
                    trajectory_seq =  []
                #########################################################################
                mujoco.mj_step(env, data)
                counter += 1
                
        start = timer()
        ref_base_lin_vel = jnp.array([0.3,0,0])
        ref_base_ang_vel = jnp.array([0,0,0.0])
        
        # x0 = jnp.concatenate([qpos, qvel,jnp.zeros(3*config.n_contact)])
        input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                        ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                        1.0])
        
        #set this to the current contact state to use the blind step adaptation
        contact = np.zeros(config.n_contact)
    
        start = timer()
        tau, q, dq  = mpc.run(qpos,qvel,input,contact)   
        stop = timer()
        
        # print(f"Time elapsed: {stop-start}")           
    counter += 1        
    data.ctrl = tau - 3*qvel[6:6+config.n_joints]
    mujoco.mj_step(env, data)
    # viewer.sync()

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
                    torch.save(running_loss, result_dir + 'loss_adam.pth')
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




