import torch
from ExplorationModule.EplorationModule import ExplorationModule
from Agent import dnn_models
import numpy as np
from utils.common import RunningStats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RND(ExplorationModule):
    def __init__(self, state_dim, lr=0.001):
        super(RND, self).__init__()
        self.predictor = dnn_models.ConvNetFeatureExtracor(state_dim, [(32, 8, 4), (64, 4, 2), (64, 3, 1), 512, 512, 512])
        with torch.no_grad():
            self.target = dnn_models.ConvNetFeatureExtracor(state_dim, [(32, 8, 4), (64, 4, 2), (64, 3, 1), 512])

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.optimizer.zero_grad()

        # Initialize weights
        for m in [x for x in self.predictor.modules()] + [x for x in self.target.modules()]:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        self.i_reward_normalizer = RunningStats((1,), computation_module=torch)
        self.state_normalizer = RunningStats(state_dim, computation_module=torch)

    def compute_intrinsic_reward(self, cur_states, next_states, actions):
        # Normalize states: as paper suggests its crucial for random networks
        next_states = torch.from_numpy(next_states).to(device).float()
        self.state_normalizer.update(next_states)
        next_states = self.state_normalizer.scale(next_states)
        next_states = torch.clamp(next_states,-5, 5).float()

        # Compute intrinsic reward and optimize
        intrinsic_rewards = 0.5*(self.target(next_states) - self.predictor(next_states))**2
        intrinsic_rewards = intrinsic_rewards.mean(1)
        loss = intrinsic_rewards.mean()
        loss.backward()
        self.optimizer.step()
        self.reporter.add_costume_log("RND-loss", None, loss.item())

        # return normalized intrinsic reward
        i_reward = intrinsic_rewards.detach()
        self.i_reward_normalizer.update(i_reward)
        i_reward = self.i_reward_normalizer.scale(i_reward, substract_mean=False)
        return i_reward

