import os
from utils import loggers
from Agent.PPO_parallel import PPOParallel
from Agent.AgentConfigs import get_agent_configs
from Enviroment.EnvBuilder import get_env_builder, get_env_goal
from train import train_agent_multi_env, train_progress_manager, test
from opt import *
from gym import wrappers
from ExplorationModule.RND import RND
from ExplorationModule.DummyExploration import DummyExplorationModule


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
    dummy_env = env_builder()
    exp_module = RND(dummy_env.observation_space.shape)
    # exp_module = DummyExplorationModule()
    agent = PPOParallel(dummy_env.observation_space, dummy_env.action_space, exp_module, hp)
    if WEIGHTS_FILE:
        agent.load_state(WEIGHTS_FILE)
    train_dir = os.path.join(TRAIN_ROOT, ENV_NAME,  agent.name)

    if TRAIN:
        logger = get_logger(LOGGER_TYPE, log_frequency=LOG_FREQUENCY, logdir=train_dir)
        agent.set_reporter(logger)
        progress_maneger = train_progress_manager(train_dir, get_env_goal(ENV_NAME), SCORE_SCOPE, logger,
                                                  checkpoint_steps=CKP_STEP, temporal_frequency=TEMPORAL_FREQ)

        # train_agent_multi_env(env_builder, agent, progress_maneger, test_frequency=TEST_FREQUENCY, save_videos=SAVE_VIDEOS)
        agent.train_agent(env_builder, progress_maneger, test_frequency=TEST_FREQUENCY, save_videos=SAVE_VIDEOS)

    else:
        # Test
        env = env_builder(test_config=True)
        env = wrappers.Monitor(env, os.path.join(train_dir, "test"),
                            video_callable=lambda episode_id: True, force=True)
        score = test(env, agent, TEST_EPISODES)
        print("Avg reward over %d episodes: %f"%(TEST_EPISODES, score))
