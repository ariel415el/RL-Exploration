import os
from utils import loggers
from Agent.PPO_parallel import PPOParallel
from Agent.AgentConfigs import get_agent_configs
from Enviroment.EnvBuilder import get_env_builder, get_env_goal
from train import train_agent_multi_env, train_progress_manager, test
from opt import *
from gym import wrappers
import gym

def build_agent(env,  hp):
    state_dim = env.observation_space.shape
    if type(env.action_space) == gym.spaces.Discrete:
        action_dim = env.action_space.n
    else:
        action_dim = [env.action_space.low, env.action_space.high]
    agent = PPOParallel(state_dim, action_dim, hp)
    return agent


def get_logger(logger_type, log_frequency, logdir):
    if logger_type == 'plt':
        constructor = loggers.plt_logger
    elif logger_type == 'tensorboard':
        constructor = loggers.TB_logger
    else:
        constructor = loggers.logger

    return constructor(log_frequency, logdir)


if __name__ == '__main__':
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)

    env_builder = get_env_builder(ENV_NAME)
    hp = get_agent_configs(ENV_NAME)
    print(hp)
    agent = build_agent(env_builder(), hp)
    if WEIGHTS_FILE:
        agent.load_state(WEIGHTS_FILE)
    train_dir = os.path.join(TRAIN_ROOT, ENV_NAME,  agent.name)

    if TRAIN:
        logger = get_logger(LOGGER_TYPE, log_frequency=LOG_FREQUENCY, logdir=train_dir)
        agent.set_reporter(logger)
        progress_maneger = train_progress_manager(train_dir, get_env_goal(ENV_NAME), SCORE_SCOPE, logger,
                                                  checkpoint_steps=CKP_STEP, temporal_frequency=TEMPORAL_FREQ)

        train_agent_multi_env(env_builder, agent, progress_maneger)

    else:
        # Test
        env = env_builder(test_config=True)
        env = wrappers.Monitor(env, os.path.join(train_dir, "test"),
                            video_callable=lambda episode_id: True, force=True)
        score = test(env, agent, TEST_EPISODES)
        print("Avg reward over %d episodes: %f"%(TEST_EPISODES, score))
