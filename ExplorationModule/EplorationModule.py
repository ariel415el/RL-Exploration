import torch

class ExplorationModule:
    def __init__(self):
        self.reporter = None

    def compute_intrinsic_reward(self, cur_states, next_states, actions):
        return torch.zeros((next_states.shape[0], next_states.shape[1], 1)).to(next_states.device())

    def set_reporter(self, reporter):
        self.reporter = reporter
