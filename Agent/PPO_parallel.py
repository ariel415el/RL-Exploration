import os
from Agent.dnn_models import *
from utils.common import *
from torch.utils.data import DataLoader
from utils.common import BasicDataset, safe_update_dict
import gym
from Enviroment.MultiEnvs import MultiEnviroment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class episodic_memory:
    def __init__(self,concurrency, horizon, state_dim, num_outputs):
        self.states = torch.zeros((concurrency, horizon) + state_dim).to(device)
        self.actions = torch.zeros((concurrency, horizon, num_outputs)).to(device)
        self.e_values = torch.zeros((concurrency, horizon + 1, 1)).to(device)
        self.i_values = torch.zeros((concurrency, horizon + 1, 1)).to(device)
        self.logprobs = torch.zeros((concurrency, horizon, 1)).to(device)
        self.e_rewards = torch.zeros((concurrency, horizon, 1)).to(device)
        self.i_rewards = torch.zeros((concurrency, horizon, 1)).to(device)
        self.is_terminals = torch.zeros((concurrency, horizon, 1)).to(device)


class PPOParallel:
    def __init__(self, observation_space, action_space, exploration_module, hp, train=True):
        self.state_space = observation_space
        self.action_space = action_space
        self.train= train
        self.reporter = None
        self.exploration_module = exploration_module
        self.hp = {
            'concurrent_epsiodes':8,
            'horizon':128,
            'epochs': 3,
            'minibatch_size':32,
            'e_discount':0.99,
            'lr':0.01,
            'lr_decay':0.995,
            'epsilon_clip':0.2,
            'fe_layers':[64],
            'model_layers':[64],
            'entropy_weight':0.01,
            'grad_clip':0.5,
            'GAE': 1, # 1 for MC, 0 for TD,
            'i_discount': 0.999,
            'i_value_coef': 0.5,
            'i_advantage_coef': 0.5
        }
        safe_update_dict(self.hp, hp)
        self.e_reward_normalizer = None

        if len(self.state_space.shape) > 1:
            feature_extractor = ConvNetFeatureExtracor(self.state_space.shape, self.hp['fe_layers'])
        else:
            feature_extractor = LinearFeatureExtracor(self.state_space.shape[0], self.hp['fe_layers'],
                                                      batch_normalization=False, activation=nn.ReLU())

        if type(self.action_space) == gym.spaces.Discrete:
            self.num_outputs = 1
            self.policy = ActorCriticModel(feature_extractor, self.action_space.n, self.hp['model_layers'],
                                           discrete=True, activation=nn.ReLU()).to(device)
        else:
            self.num_outputs = self.action_space.shape[0]
            self.policy = ActorCriticModel(feature_extractor, self.num_outputs, self.hp['model_layers'],
                                           discrete=False, activation=nn.ReLU()).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.hp['lr'])
        self.optimizer.zero_grad()

        self.num_steps = 0
        self.learn_steps = 0

        self.memory = episodic_memory(self.hp['concurrent_epsiodes'], self.hp['horizon'], self.state_space.shape, self.num_outputs)

        self.name = 'PPO-Parallel'
        self.name += "_lr[%.5f]_b[%d]_GAE[%.2f]_ec[%.1f]_l-%s-%s"%(
            self.hp['lr'], self.hp['concurrent_epsiodes'], self.hp['GAE'], self.hp['epsilon_clip'], str(self.hp['fe_layers']), str(self.hp['model_layers']))
        if  self.hp['grad_clip'] is not None:
            self.name += "_gc[%.1f]"%self.hp['grad_clip']

    def load_state(self, path):
        if os.path.exists(path):
            # if trained on gpu but test on cpu use:
            self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def save_state(self, path):
        torch.save(self.policy.state_dict(), path)

    def process_states(self, states):
        torch_states = torch.from_numpy(states).to(device).float()
        dists, e_values, i_values = self.policy(torch_states)
        if self.num_steps == self.hp['horizon']:
            self.memory.e_values[:, self.num_steps] = e_values.detach()
            self.memory.i_values[:, self.num_steps] = i_values.detach()
            self._learn()
            self.learn_steps += 1

            if (self.learn_steps+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.hp['lr_decay']
                self.reporter.add_costume_log("lr", self.learn_steps, self.optimizer.param_groups[0]['lr'])

            self.num_steps = 0

        actions = dists.sample().detach()
        self.memory.states[:, self.num_steps] = torch_states
        self.memory.actions[:, self.num_steps] = actions.view(-1, self.num_outputs)
        self.memory.e_values[:, self.num_steps] = e_values.detach()
        self.memory.i_values[:, self.num_steps] = i_values.detach()
        self.memory.logprobs[:, self.num_steps] = dists.log_prob(actions).detach().view(-1,1)

        return self._get_output_actions(actions)

    def update_step_results(self, e_rewards, i_rewards, is_next_state_terminals):
        # Normalize extrinsic rewards
        e_rewards = torch.from_numpy(e_rewards).to(device)
        self.memory.e_rewards[:, self.num_steps] = e_rewards.view(-1, 1)
        self.memory.i_rewards[:, self.num_steps] = i_rewards.view(-1, 1)
        self.memory.is_terminals[:, self.num_steps] = torch.from_numpy(is_next_state_terminals).to(device).view(-1, 1)
        self.num_steps += 1
        self.reporter.add_costume_log("Intrinscir Reward",None, i_rewards.mean().item())
        self.reporter.add_costume_log("Extrinsic Reward",None, e_rewards.mean().item())

    def _get_output_actions(self, actions):
        output_actions = actions.detach().cpu().numpy() # Using only this is problematic for super mario since it returns a 0-size np array in discrete action space
        if type(self.action_space) != gym.spaces.Discrete:
            output_actions = np.clip(output_actions, self.action_space.low, self.action_space.high)

        return output_actions

    def _create_lerning_data(self):
        # Compute extrinsic advantages and rewards
        cur_e_values = self.memory.e_values[:,:-1]
        next_e_values = self.memory.e_values[:,1:]
        deltas = self.memory.e_rewards + self.hp['e_discount'] * next_e_values * (1 - self.memory.is_terminals) - cur_e_values
        advantages = discount_batch(deltas, self.memory.is_terminals, self.hp['GAE'] * self.hp['e_discount'], device)
        e_rewards = discount_batch(self.memory.e_rewards, self.memory.is_terminals,  self.hp['e_discount'], device)
        e_advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)

        # Compute intrixsic advantages and rewards
        cur_i_values = self.memory.i_values[:,:-1]
        next_i_values = self.memory.i_values[:,1:]
        deltas = self.memory.i_rewards + self.hp['i_discount'] * next_i_values * (1 - self.memory.is_terminals) - cur_i_values
        advantages = discount_batch(deltas, self.memory.is_terminals, self.hp['GAE'] * self.hp['i_discount'], device)
        i_rewards = discount_batch(self.memory.i_rewards, self.memory.is_terminals,  self.hp['i_discount'], device)
        i_advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)

        advantages = e_advantages + self.hp['i_advantage_coef']*i_advantages

        # Create a dataset from flatten data
        dataset = BasicDataset(self.memory.states.view(-1, *self.memory.states.shape[2:]),
                               cur_e_values.reshape(-1, *cur_e_values.shape[2:]),  # view not working here (reshape copeies)
                               cur_i_values.reshape(-1, *cur_i_values.shape[2:]),  # view not working here (reshape copeies)
                               self.memory.actions.view(-1, *self.memory.actions.shape[2:]),
                               self.memory.logprobs.view(-1, *self.memory.logprobs.shape[2:]),
                               e_rewards.view(-1, *e_rewards.shape[2:]),
                               i_rewards.view(-1, *i_rewards.shape[2:]),
                               advantages.view(-1, *advantages.shape[2:]))
        dataloader = DataLoader(dataset, batch_size=self.hp['minibatch_size'], shuffle=True)

        return dataloader

    def _learn(self):
        dataloader = self._create_lerning_data()

        for _ in range(self.hp['epochs']):
            for (states_batch, old_policy_e_values_batch, old_policy_i_values_batch, old_policy_actions_batch,
                 old_policy_loggprobs_batch, e_rewards_batch, i_rewards_batch, advantages_batch) in dataloader:
                # Evaluating old actions and values with the target policy:
                dists, e_values, i_values = self.policy(states_batch)
                exploration_loss = -self.hp['entropy_weight'] * dists.entropy()
                critic_loss = 0.5 * (e_values - e_rewards_batch).pow(2) + \
                              self.hp['i_value_coef'] * 0.5 * (i_values - i_rewards_batch).pow(2)

                # Finding the ratio (pi_theta / pi_theta_old):
                # logprobs = dists.log_prob(old_policy_actions_batch.view(-1))
                if self.num_outputs == 1:
                    logprobs = dists.log_prob(old_policy_actions_batch.view(-1))
                else:
                    logprobs = dists.log_prob(old_policy_actions_batch)
                ratios = torch.exp(logprobs.view(-1,1) - old_policy_loggprobs_batch)
                # Finding Surrogate actor Loss:
                surr1 = advantages_batch * ratios
                surr2 = advantages_batch * torch.clamp(ratios, 1 - self.hp['epsilon_clip'],
                                                       1 + self.hp['epsilon_clip'])
                actor_loss = -torch.min(surr1, surr2)

                loss = actor_loss.mean() + critic_loss.mean() + exploration_loss.mean()
                loss.backward()

                if self.hp['grad_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hp['grad_clip'])
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.reporter.add_costume_log("actor_loss", None, actor_loss.mean().item())
                self.reporter.add_costume_log("critic_loss", None, critic_loss.mean().item())
                self.reporter.add_costume_log("dist_entropy", None, -exploration_loss.mean().item())
                self.reporter.add_costume_log("ratios", None, ratios.mean().item())
                self.reporter.add_costume_log("e_values", None, e_values.mean().item())
                self.reporter.add_costume_log("i_values", None, i_values.mean().item())

    def set_reporter(self, reporter):
        self.reporter = reporter
        if self.exploration_module:
            self.exploration_module.set_reporter(reporter)

    def train_agent(self, env_builder, progress_manager, test_frequency, save_videos):
        """Train agent that can train with multiEnv objects"""
        # self.exploration_module.init(env_builder)
        running_scores = [0 for _ in range(self.hp['concurrent_epsiodes'])]
        running_lengths = [0 for _ in range(self.hp['concurrent_epsiodes'])]

        multi_env = MultiEnviroment(env_builder, self.hp['concurrent_epsiodes'])
        states = multi_env.get_initial_state()

        while not progress_manager.training_complete:
            actions = self.process_states(states)
            next_states, e_rewards, is_next_state_terminals, infos = multi_env.step(actions)
            i_reward = self.exploration_module.compute_intrinsic_reward(states, next_states, actions)
            self.update_step_results(e_rewards, i_reward, is_next_state_terminals)
            states = next_states

            for i, (reward, done) in enumerate(zip(e_rewards, is_next_state_terminals)):
                running_scores[i] += reward
                running_lengths[i] += 1
                if done:
                    save_path = progress_manager.report_episode(running_scores[i], running_lengths[i])
                    if save_path is not None:
                        self.save_state(save_path)
                    running_scores[i] = 0
                    running_lengths[i] = 0

                if (progress_manager.episodes_done + 1) % test_frequency == 0:
                    from train import test
                    from gym import wrappers
                    test_env = env_builder(test_config=True)
                    if save_videos:
                        test_env = gym.wrappers.Monitor(test_env, os.path.join(progress_manager.videos_dir, "test_%d" % (
                                    progress_manager.episodes_done + 1)), video_callable=lambda episode_id: True,
                                                        force=True)
                    test_score = test(test_env, self, 1)

                    progress_manager.report_test(test_score)
                    test_env.close()

        multi_env.close()