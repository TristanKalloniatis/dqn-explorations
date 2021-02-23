import numpy as np
import gym


def equal(state, goal_state):
    return (state == goal_state).all()


class GoalEnv:
    def __init__(self, n, shaped_reward):
        self.n = n
        self.shaped_reward = shaped_reward

        self.state = None
        self.goal_state = None
        self.action_space = []
        self.episode_length = 0
        self.max_episode_length = 1
        self.store_goals = True
        self.image_observations = False

        self.non_trivial_state()

    @property
    def num_actions(self):
        return len(self.action_space)

    def generate_state(self):
        raise NotImplementedError

    def non_trivial_state(self):
        self.state = self.generate_state()
        self.goal_state = self.generate_state()
        while self.solved:
            self.non_trivial_state()

    @property
    def solved(self):
        return equal(self.state, self.goal_state)

    def compute_done(self, state, goal_state):
        return (
            equal(state, goal_state) or self.episode_length == self.max_episode_length
        )

    @property
    def done(self):
        return self.compute_done(self.state, self.goal_state)

    @property
    def info(self):
        return {}

    def compute_observation(self, state):
        raise NotImplementedError

    def compute_state(self, observation):
        raise NotImplementedError

    @property
    def observation(self):
        return self.compute_observation(self.state)

    @property
    def goal(self):
        return self.compute_observation(self.goal_state)

    def compute_reward(self, state, goal_state):
        if self.shaped_reward:
            return -np.mean((state - goal_state) ** 2)
        else:
            return 0 if self.solved else -1

    def compute_reward_from_observation(self, observation, goal):
        state = self.compute_state(observation)
        goal_state = self.compute_state(goal)
        return self.compute_reward(state, goal_state)

    @property
    def reward(self):
        return self.compute_reward(self.state, self.goal_state)

    def valid_action(self, action):
        return action in self.action_space

    def compute_step(self, action):
        raise NotImplementedError

    def step(self, action):
        assert action in self.action_space, "Invalid action {0}".format(action)
        self.compute_step(action)
        self.episode_length += 1
        return self.observation, self.reward, self.done, self.info

    def reset(self):
        self.__init__(self.n, self.shaped_reward)
        return self.observation


class BitFlipper(GoalEnv):
    def __init__(self, n, shaped_reward=False):
        self.n = n

        super().__init__(n, shaped_reward)

        self.action_space = range(n)
        self.max_episode_length = n

    def generate_state(self):
        return np.random.randint(low=0, high=2, size=self.n)

    def compute_observation(self, state):
        return np.array(state, dtype=np.float32)

    def compute_state(self, observation):
        return np.array(observation, dtype=np.int64)

    def compute_step(self, action):
        self.state[action] = 1 - self.state[action]


class GridWorld(GoalEnv):
    def __init__(self, n, shaped_reward=False, image_observations=True):
        self.n = n

        super().__init__(n, shaped_reward)

        self.image_observations = image_observations
        self.action_space = range(4)
        self.max_episode_length = 2 * n

    def generate_state(self):
        return np.random.randint(low=0, high=self.n, size=2)

    def compute_observation(self, state):
        if self.image_observations:
            observation = np.zeros((self.n, self.n), dtype=np.float32)
            observation[tuple(state)] = 1
            observation = np.expand_dims(
                observation, axis=0
            )  # Insert a channel dimension
        else:
            observation = np.array(state, dtype=np.float32)
        return observation

    def compute_state(self, observation):
        if self.image_observations:
            observation = observation.squeeze(0)
            state = np.array(
                [
                    np.argmax(observation, axis=0).sum(),
                    np.argmax(observation, axis=1).sum(),
                ],
                dtype=np.int64,
            )
        else:
            state = np.array(observation, dtype=np.int64)
        return state

    def compute_step(self, action):
        if action == 0:
            self.state[0] = max(self.state[0] - 1, 0)
        elif action == 1:
            self.state[1] = max(self.state[1] - 1, 0)
        elif action == 2:
            self.state[0] = min(self.state[0] + 1, self.n - 1)
        elif action == 3:
            self.state[1] = min(self.state[1] + 1, self.n - 1)

    def reset(self):
        self.__init__(self.n, self.shaped_reward, self.image_observations)
        return self.observation

    def plot(self):  # Purely for debugging purposes
        from matplotlib.pyplot import imshow

        imshow((self.observation - self.goal).squeeze(0))

    # Semantic actions for debugging purposes:

    def up(self):
        return self.step(0)

    def left(self):
        return self.step(1)

    def down(self):
        return self.step(2)

    def right(self):
        return self.step(3)


class GymEnv:
    def __init__(self, name):
        self.env = gym.make(name)
        self.n = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.observation = self.reset()
        self.store_goals = False
        self.image_observations = False

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()
