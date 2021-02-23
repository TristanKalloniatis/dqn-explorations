import torch
import numpy as np
from math import exp
from buffer import ReplayBuffer
from networks import DuellingDQN
import environments
import random
from log_utils import create_logger, write_log
import hyperparameters

log_file = "train"
logger = create_logger(log_file)
use_cuda = torch.cuda.is_available()
message = "Found GPU {0}".format(torch.device("cuda")) if use_cuda else "No GPU"
write_log(message, logger)


def epsilon_by_frame(num_frames_seen, epsilon_start, epsilon_end, epsilon_decay_rate):
    return epsilon_end + (epsilon_start - epsilon_end) * exp(
        -num_frames_seen / epsilon_decay_rate
    )


def build_environment(env_config):
    name = env_config["name"]
    if name == "bit":
        n = env_config["n"] if "n" in env_config else hyperparameters.DEFAULT_SIZE
        shaped_reward = (
            env_config["shaped_reward"] if "shaped_reward" in env_config else False
        )
        return environments.BitFlipper(n, shaped_reward)
    elif name == "grid":
        n = env_config["n"] if "n" in env_config else hyperparameters.DEFAULT_SIZE
        shaped_reward = (
            env_config["shaped_reward"] if "shaped_reward" in env_config else False
        )
        image_observations = (
            env_config["image_observations"]
            if "image_observations" in env_config
            else True
        )
        return environments.GridWorld(n, shaped_reward, image_observations)
    else:
        return environments.GymEnv(name)


class DQN:
    def __init__(
        self,
        env_name,
        env_params={},
        buffer_size=hyperparameters.BUFFER_SIZE,
        batch_size=hyperparameters.BATCH_SIZE,
        hidden_size=hyperparameters.HIDDEN_SIZE,
        act=torch.nn.ReLU,
        epsilon_start=hyperparameters.EPSILON_START,
        epsilon_end=hyperparameters.EPSILON_END,
        epsilon_decay_rate=hyperparameters.EPSILON_DECAY_RATE,
        evaluate_episodes=hyperparameters.EVALUATE_EPISODES,
        use_hindsight=False,
        polyak_weight=hyperparameters.POLYAK_WEIGHT,
        smoothing_factor=hyperparameters.SMOOTHING_FACTOR,
        separate_goals=True,
        gamma=hyperparameters.GAMMA,
        loss_function=torch.nn.MSELoss,
        log_freq=hyperparameters.LOG_FREQ,
    ):
        env_config = env_params
        env_config["name"] = env_name
        write_log("DQN config", logger)
        write_log("env_config: {0}".format(env_config), logger)
        write_log("buffer_size: {0}".format(buffer_size), logger)
        write_log("batch_size: {0}".format(batch_size), logger)
        write_log("hidden_size: {0}".format(hidden_size), logger)
        write_log("act: {0}".format(act), logger)
        write_log("epsilon_start: {0}".format(epsilon_start), logger)
        write_log("epsilon_end: {0}".format(epsilon_end), logger)
        write_log("epsilon_decay_rate: {0}".format(epsilon_decay_rate), logger)
        write_log("evaluate_episodes: {0}".format(evaluate_episodes), logger)
        write_log("use_hindsight: {0}".format(use_hindsight), logger)
        write_log("polyak_weight: {0}".format(polyak_weight), logger)
        write_log("smoothing_factor: {0}".format(smoothing_factor), logger)
        write_log("separate_goals: {0}".format(separate_goals), logger)
        write_log("gamma: {0}".format(gamma), logger)
        write_log("loss_function: {0}".format(loss_function), logger)
        write_log("log_freq: {0}".format(log_freq), logger)

        self.evaluate_episodes = evaluate_episodes
        self.use_hindsight = use_hindsight
        self.polyak_weight = polyak_weight
        self.act = act
        self.separate_goals = separate_goals
        self.gamma = gamma
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.log_freq = log_freq
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.smoothing_factor = smoothing_factor

        self.environment = build_environment(env_config)
        self.buffer = ReplayBuffer(
            buffer_size, batch_size, self.environment.store_goals
        )
        self.current_network = DuellingDQN(
            self.environment.n,
            self.environment.image_observations,
            self.environment.num_actions,
            hidden_size,
            act,
            separate_goals=self.environment.store_goals and separate_goals,
        )
        self.target_network = DuellingDQN(
            self.environment.n,
            self.environment.image_observations,
            self.environment.num_actions,
            hidden_size,
            act,
            separate_goals=self.environment.store_goals and separate_goals,
        )
        self.synchronise(False)

        self.optimiser = torch.optim.Adam(self.current_network.parameters())
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            self.current_network.cuda()
            self.target_network.cuda()
        self.episode_rewards = []
        self.losses = []
        self.smoothed_episode_rewards = []
        self.smoothed_losses = []
        self.num_episodes_trained = 0
        self.num_frames_seen = 0

    @property
    def epsilon(self):
        return epsilon_by_frame(
            self.num_frames_seen,
            self.epsilon_start,
            self.epsilon_end,
            self.epsilon_decay_rate,
        )

    def synchronise(self, use_polyak=True):
        weight = self.polyak_weight if use_polyak else 1.0
        for x, y in zip(
            self.current_network.parameters(), self.target_network.parameters()
        ):
            y.data.copy_(weight * x.data + (1 - weight) * y.data)

    def train_on_batch(self):
        goals = None
        if self.environment.store_goals:
            (
                observations,
                actions,
                rewards,
                next_observations,
                dones,
                goals,
            ) = self.buffer.sample()
        else:
            (
                observations,
                actions,
                rewards,
                next_observations,
                dones,
            ) = self.buffer.sample()
        observations = torch.tensor(observations, device=self.device).float()
        actions = torch.tensor(actions, device=self.device).long()
        rewards = torch.tensor(rewards, device=self.device).float()
        next_observations = torch.tensor(next_observations, device=self.device).float()
        dones = torch.tensor(dones, device=self.device).bool()
        if self.environment.store_goals:
            goals = torch.tensor(goals, device=self.device).float()
        current_values = (
            self.current_network(observations, goals)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )
        with torch.no_grad():
            next_values = self.target_network(next_observations, goals).max(-1)[0]
            targets = rewards + torch.where(
                dones, torch.zeros_like(rewards), self.gamma * next_values
            )
        loss = self.loss_function()(current_values, targets)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.losses.append(loss.item())
        if len(self.smoothed_losses) == 0:
            self.smoothed_losses.append(loss.item())
        else:
            self.smoothed_losses.append(
                self.smoothing_factor * self.smoothed_losses[-1]
                + (1 - self.smoothing_factor) * loss.item()
            )

    def play_episode(self, training=True):
        observation = self.environment.reset()
        done = False
        episode_reward = 0
        if training:
            observations = [observation]
            actions = []
            rewards = []
            dones = []
        while not done:
            if training and np.random.rand() < self.epsilon:
                action = random.choice(range(self.environment.num_actions))
            else:
                observation = observation.copy()
                observation_tensor = (
                    torch.tensor(observation, device=self.device).float().unsqueeze(0)
                )
                goal_tensor = (
                    torch.tensor(self.environment.goal, device=self.device)
                    .float()
                    .unsqueeze(0)
                    if self.environment.store_goals
                    else None
                )
                with torch.no_grad():
                    action = (
                        self.current_network(observation_tensor, goal_tensor)
                        .argmax(-1)
                        .item()
                    )
            next_observation, reward, done, _ = self.environment.step(action)
            episode_reward += reward
            observation = next_observation.copy()
            if training:
                self.num_frames_seen += 1
                observations.append(observation.copy())
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                if len(self.buffer) > self.batch_size:
                    self.train_on_batch()
                    self.synchronise()
        if training:
            self.num_episodes_trained += 1
            self.episode_rewards.append(episode_reward)
            if len(self.smoothed_episode_rewards) == 0:
                self.smoothed_episode_rewards.append(episode_reward)
            else:
                self.smoothed_episode_rewards.append(
                    self.smoothing_factor * self.smoothed_episode_rewards[-1]
                    + (1 - self.smoothing_factor) * episode_reward
                )
            for i in range(len(actions)):
                if self.environment.store_goals:
                    self.buffer.add(
                        (
                            observations[i],
                            actions[i],
                            rewards[i],
                            observations[i + 1],
                            dones[i],
                            self.environment.goal,
                        )
                    )
                    if self.use_hindsight:
                        goal = observations[-1]
                        done = i == len(actions) - 1
                        self.buffer.add(
                            (
                                observations[i],
                                actions[i],
                                self.environment.compute_reward_from_observation(
                                    observations[i + 1],
                                    goal,
                                ),
                                observations[i + 1],
                                done,
                                goal,
                            )
                        )
                else:
                    self.buffer.add(
                        (
                            observations[i],
                            actions[i],
                            rewards[i],
                            observations[i + 1],
                            dones[i],
                        )
                    )
        elif self.environment.store_goals:
            return self.environment.solved

    def train(self, episodes=hyperparameters.TRAIN_EPISODES):
        while self.num_episodes_trained < episodes:
            self.play_episode()
            if self.num_episodes_trained % self.log_freq == 0:
                write_log(
                    "Trained episode {0}/{1}, latest smoothed reward {2}".format(
                        self.num_episodes_trained,
                        episodes,
                        self.smoothed_episode_rewards[-1],
                    ),
                    logger,
                )

    def evaluate(self, episodes=hyperparameters.EVALUATE_EPISODES):
        write_log("Evaluating on {0} episodes".format(episodes), logger)
        success_rate = np.mean([self.play_episode(False) for _ in range(episodes)])
        write_log("Result: {0}".format(success_rate), logger)
        return success_rate
