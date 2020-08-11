"""
This file contain functions that returns enviroment factories and agent with according configurations
"""

def get_agent_configs(env_name):
    if env_name == "CartPole-v1":
        agent_configs  = {'lr': 0.001, 'lr_decay': 0.9999, 'concurrent_epsiodes': 4, 'horizon': 128, 'epochs': 3,
                        'minibatch_size': 32, 'GAE': 0.95, 'epsilon_clip': 0.1, 'grad_clip': 0.5, 'fe_layers': [64],
                        'model_layers': [32, 32]}

    elif env_name == "MountainCar-v0":
        agent_configs = {'lr': 0.002, 'lr_decay': 0.9999, 'concurrent_epsiodes': 16, 'horizon': 256, 'epochs': 3,
                            'minibatch_size': 32, 'GAE': 0.95, 'epsilon_clip': 0.2, 'grad_clip': 0.5,
                            'fe_layers': [64, 64], 'model_layers': [],
                            }

    elif env_name == "Pendulum-v0":
        agent_configs = {'lr': 0.001, 'lr_decay': 0.99, 'concurrent_epsiodes': 8, 'horizon': 128, 'epochs': 3,
                         'minibatch_size': 32, 'GAE': 0.95, 'epsilon_clip': 0.1, 'grad_clip': 0.5,
                         'fe_layers': [64], 'model_layers': []}

    elif env_name == "LunarLander-v2":
        agent_configs = {'lr': 0.001, 'lr_decay': 0.99, 'concurrent_epsiodes': 8, 'horizon': 128, 'epochs': 3,
                            'minibatch_size': 32,'GAE': 0.95, 'epsilon_clip': 0.1, 'grad_clip': 0.5,
                             'fe_layers': [64], 'model_layers': [32, 32]}


    elif env_name == "BipedalWalker-v3":
        agent_configs = {'concurrent_epsiodes': 1, 'horizon': 2048, 'epochs': 10, 'minibatch_size': 64,
                         'lr': 0.0003, 'lr_decay': 0.995,'GAE': 0.95, 'epsilon_clip': 0.2, 'grad_clip': 0.5, 'entropy_weight':0.0,
                         'fe_layers': [], "model_layers":[64,64]}

    elif "MontezumaRevenge" in env_name:
        agent_configs = {'lr': 0.0001, 'lr_decay': 0.999, 'concurrent_epsiodes': 128, 'epochs': 3, 'e_discount':0.999,
                        'minibatch_size': 32, 'GAE': 0.95, 'epsilon_clip': 0.1,  'grad_clip': 0.5, 'entropy_weight': 0.01,
                         'fe_layers': [(32, 8, 4), (64, 4, 2), (64, 3, 1), 512, 512], 'model_layers': [],
                         'i_value_coef':0.0, 'i_advantage_coef':0.0, 'i_discount':0.99}

    elif "NoFrameskip" in env_name:
        agent_configs = {'lr': 0.00025, 'lr_decay': 0.999, 'concurrent_epsiodes': 16, 'epochs': 3,
                        'minibatch_size': 32, 'GAE': 0.95, 'epsilon_clip': 0.1,  'grad_clip': 0.5, 'entropy_weight': 0.01,
                         'fe_layers': [(32, 8, 4), (64, 4, 2), (64, 3, 1), 512, 512], 'model_layers': [],
                         'i_value_coef':0.0, 'i_advantage_coef':0.0}

    elif env_name == "HalfCheetahMuJoCoEnv-v0":
        agent_configs = {'concurrent_epsiodes': 1, 'horizon': 2048, 'epochs': 10, 'minibatch_size': 64,
                        'GAE': 0.95, 'epsilon_clip': 0.2, 'grad_clip': 0.5, 'entropy_weight':0.0, 'lr': 0.0003,'lr_decay': 0.995,
                        'fe_layers': [], "model_layers":[64,64]}

    elif "SuperMarioBros" in env_name:
        agent_configs = {'lr': 0.00025, 'lr_decay': 0.9999, 'concurrent_epsiodes': 16, 'epochs': 3,
                         'minibatch_size': 32, 'GAE': 0.95,'epsilon_clip': 0.1, 'grad_clip': 0.5, 'entropy_weight': 0.01,
                         'fe_layers': [(32, 8, 4), (64, 4, 2), (64, 3, 1), 512, 512], 'model_layers': []}

    elif "MiniGrid" in env_name:
        agent_configs = {'lr': 0.00025, 'lr_decay': 0.9999, 'concurrent_epsiodes': 16, 'epochs': 3,
                         'minibatch_size': 32, 'GAE': 0.95,'epsilon_clip': 0.1, 'grad_clip': 0.5, 'entropy_weight': 0.01,
                         'fe_layers': [(32, 8, 4), (64, 4, 2), (64, 3, 1), 512, 512], 'model_layers': []}
    else:
        agent_configs = {}

    return agent_configs
