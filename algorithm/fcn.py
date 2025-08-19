import numpy as np

def flatten_state(state):
    return np.concatenate([np.ravel(v) for v in state.values()])