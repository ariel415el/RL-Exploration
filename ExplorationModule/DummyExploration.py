import torch
from ExplorationModule.EplorationModule import ExplorationModule
from Agent import dnn_models
import numpy as np
from utils.common import RunningStats

class DummyExplorationModule(ExplorationModule):
    def __init__(self):
        super(DummyExplorationModule, self).__init__()

    def compute_intrinsic_reward(self, cur_states, next_states, actions):
        return torch.zeros(actions.shape)

