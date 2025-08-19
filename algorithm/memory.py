import random
from collections import namedtuple, deque





class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, trajectory_seq):
        """Save a trajectory"""
        self.memory.append(trajectory_seq)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)





# Trajectory = namedtuple('Trajectory', ('trajectory_seq'))

# class ReplayMemory(object):

#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)

#     def push(self, trajectory_seq):
#         """Save a trajectory"""
#         self.memory.append(Trajectory(trajectory_seq))

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)