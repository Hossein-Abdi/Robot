import numpy as np

def flatten_state(state):
    return np.concatenate([np.ravel(v) for v in state.values()])

def reward(init_base_pos, target_base_pos):
    return 100. - (target_base_pos - init_base_pos)**2

def reward_talos(init_base_pos, target_base_pos):
    return 10*(init_base_pos - target_base_pos)