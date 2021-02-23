import random
import numpy as np
import hyperparameters


class ReplayBuffer:
    def __init__(
        self,
        buffer_size=hyperparameters.BUFFER_SIZE,
        batch_size=hyperparameters.BATCH_SIZE,
        store_goals=False,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.store_goals = store_goals

        self.buffer = []
        self.buffer_write_position = 0

    def add(self, experience):
        if len(self) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.buffer_write_position] = experience
        self.buffer_write_position = (self.buffer_write_position + 1) % self.buffer_size

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

    def __len__(self):
        return len(self.buffer)
