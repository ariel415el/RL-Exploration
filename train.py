import gym
import os
import numpy as np
from time import time
from collections import deque
from Enviroment.MultiEnvs import MultiEnviroment, MultiEnviromentSync
from gym import wrappers

class train_progress_manager(object):
    """This object is responsible of monitoring train progress, logging results"""
    def __init__(self, train_dir, solved_score, score_scope, logger, checkpoint_steps=0.2, temporal_frequency=60**2):
        self.train_dir = train_dir
        self.solved_score = solved_score
        self.checkpoint_steps = checkpoint_steps
        self.temporal_frequency = temporal_frequency
        self.logger = logger
        self.score_scope = deque(maxlen=score_scope)
        self.ckp_dir = os.path.join(train_dir, 'checkpoints')
        self.videos_dir = os.path.join(train_dir, 'videos')
        self.training_complete = False
        self.next_progress_checkpoint = 1
        self.next_time_checkpoint = 1
        self.episodes_done = 0
        self.start_time = time()
        os.makedirs(self.ckp_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

    def report_episode(self, episode_score, episode_length):
        self.score_scope.append(episode_score)
        score_scope_avg = np.mean(self.score_scope)
        self.logger.log_episode(episode_score, score_scope_avg,  episode_length)

        time_passed = time() - self.start_time
        save_path = None
        if score_scope_avg > self.next_progress_checkpoint * self.checkpoint_steps * self.solved_score:
            save_path = os.path.join(self.ckp_dir, "progress_ckp-_%.5f.pt" % score_scope_avg)
            self.next_progress_checkpoint += 1
        elif time_passed > self.temporal_frequency * self.next_time_checkpoint:
            save_path =  os.path.join(self.ckp_dir,"time_ckp_%.3f.pt"%(time_passed/360))
            self.next_time_checkpoint += 1
        self.episodes_done += 1
        if score_scope_avg >= self.solved_score:
            print("Solved in %d episodes" % self.episodes_done)
            self.training_complete = True

        return save_path


def train_agent_multi_env(env_builder, agent, progress_manager):
    """Train agent that can train with multiEnv objects"""
    multi_env = MultiEnviroment(env_builder, agent.hp['concurrent_epsiodes'])
    total_scores = [0 for _ in range(agent.hp['concurrent_epsiodes'])]
    total_lengths = [0 for _ in range(agent.hp['concurrent_epsiodes'])]
    states = multi_env.get_initial_state()
    while not progress_manager.training_complete:
        actions = agent.process_states(states)
        next_states, rewards, is_next_state_terminals, infos = multi_env.step(actions)
        agent.update_step_results(next_states, rewards, is_next_state_terminals)
        states = next_states

        for i, (reward, done) in enumerate(zip(rewards, is_next_state_terminals)):
            total_scores[i] += reward
            total_lengths[i] += 1
            if done:
                save_path = progress_manager.report_episode(total_scores[i], total_lengths[i])
                if save_path is not None:
                    agent.save_state(save_path)
                total_scores[i] = 0
                total_lengths[i] = 0

    multi_env.close()


def run_episode(env, agent):
    """Runs a full episode of a regular gym enviroment"""
    episode_rewards = []
    done = False
    state = env.reset()
    while not done:
        action = agent.process_states(np.array([state]))
        state, reward, done, info = env.step(action[0])
        episode_rewards += [reward]
    return episode_rewards


def test(env,  actor, test_episodes=1):
    actor.train = False
    episodes_total_rewards = []
    for i in range(test_episodes):
        episode_scores = run_episode(env, actor)
        env.close()
        episodes_total_rewards += [np.sum(episode_scores)]
    score = np.mean(episodes_total_rewards)
    actor.train = True
    return score
