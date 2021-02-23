import abc
import torch


class DuellingDQN(torch.nn.Module, abc.ABC):
    def __init__(
        self, num_inputs, image_inputs, num_outputs, hidden_size, act, separate_goals
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.image_inputs = image_inputs
        self.num_outputs = num_outputs
        self.hidden_size = hidden_size
        self.act = act
        self.separate_goals = separate_goals

        self.feature_map = (
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=1, out_channels=2, kernel_size=3, padding=1
                ),
                torch.nn.Flatten(),
                act(),
                torch.nn.Linear(2 * num_inputs * num_inputs, hidden_size),
                act(),
            )
            if image_inputs
            else torch.nn.Sequential(torch.nn.Linear(num_inputs, hidden_size), act())
        )

        if separate_goals:
            self.combine_features = torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_size, hidden_size), act()
            )
        self.advantage_map = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            act(),
            torch.nn.Linear(hidden_size, num_outputs),
        )
        self.value_map = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            act(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs, goals=None):
        if self.separate_goals:
            assert goals is not None, "Need to pass goals explicitly"
        elif goals is not None:
            inputs = inputs - goals
        features = self.feature_map(inputs)
        if self.separate_goals:
            goal_features = self.feature_map(goals)
            features = torch.cat([features, goal_features], dim=-1)
            features = self.combine_features(features)
        advantages = self.advantage_map(features)
        values = self.value_map(features)
        return values + advantages - advantages.mean()
