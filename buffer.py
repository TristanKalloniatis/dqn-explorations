import random
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, store_goals=False):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.store_goals = store_goals

        self.buffer = []
        self.write_position = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        if len(self) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.write_position] = experience
        self.write_position = (self.write_position + 1) % self.buffer_size

    def sample(self, size=None):
        if size is None:
            size = self.batch_size
        samples = random.sample(self.buffer, size)
        if self.store_goals:
            observations, actions, rewards, next_observations, dones, goals = zip(
                *samples
            )
            return (
                np.stack(observations),
                np.stack(actions),
                np.stack(rewards),
                np.stack(next_observations),
                np.stack(dones),
                np.stack(goals),
            )
        else:
            observations, actions, rewards, next_observations, dones = zip(*samples)
            return (
                np.stack(observations),
                np.stack(actions),
                np.stack(rewards),
                np.stack(next_observations),
                np.stack(dones),
            )
