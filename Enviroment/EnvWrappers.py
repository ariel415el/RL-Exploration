#####################################################################
# Inspires from https://github.com/higgsfield/RL-Adventure.git
#####################################################################
import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2

cv2.ocl.setUseOpenCL(False)


class RestrictAtariActions(gym.Wrapper):
    """
    Restrict atari actions to the first 5 basic actions only: 'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN'
    """
    def __init__(self, env):
        # atari_actions = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
        #                  'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE',
        #                  'DOWNLEFTFIRE']
        """Disalbe to possiblity to play the NOOP action in atari games"""
        gym.Wrapper.__init__(self, env)
        self.action_space = spaces.Discrete(5)

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env, is_atari=True):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.is_atari = is_atari
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        if self.is_atari:
            lives = self.env.unwrapped.ale.lives()
        else: # For mariobros
            lives = info['life']
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        self.lives = self.env.unwrapped.ale.lives()
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, info = self.env.step(0)
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(1, self.height, self.width), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[None, :, :]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class FireAtLostLife(gym.Wrapper):
    """
        This wrapper ensures FIRE it the first action after each lost of life which is necessary to play the game
        This is crucial when the training is done in episodic life with where there is no need to continue after lost life
         but test is on full game
    """
    def __init__(self, env):
        self.lives = 0
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives:
            obs, reward, done, info = self.env.step(1) # hit Fire
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


def get_atari_env(env_name, episode_life=False, clip_rewards=False, frame_stack=1, scale=False, simple_actions=False,
                  fire_on_lost_life=False):
    """Stack all the wrappers relavant for Atari games in the right order
        May be problematic to chane order of wraping
    """
    env = gym.make(env_name)
    assert "NoFrameskip" in  env_name # gym Deterministic skips frame by itself so pay attensions
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if fire_on_lost_life:
        env = FireAtLostLife(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, frame_stack)
    if simple_actions:
        env = RestrictAtariActions(env)
    return env


def get_super_mario_env(env_name, simple_actions=True):
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
    env = gym_super_mario_bros.make(env_name)

    # env = EpisodicLifeEnv(env, is_atari=False)
    env = WarpFrame(env)

    if simple_actions:
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
    else:
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
    return env


class channels_first(gym.core.ObservationWrapper):
    def __init__(self, env):
        gym.core.ObservationWrapper.__init__(self, env)
        shape = self.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shape[2], shape[0], shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return observation.transpose(2,0,1)

def get_grid_maze_env(env_name, image_obs=True):
    import gym_minigrid
    env = gym.make(env_name) # state is observable 7x7
    if image_obs:
        env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)  # Get pixel (image) observations instead of categorical 7x7
    env = gym_minigrid.wrappers.OneHotPartialObsWrapper(env)  # Get rid of the 'mission' field
    env = gym_minigrid.wrappers.ImgObsWrapper(env)  # Get rid of the 'mission' field
    env = channels_first(env)
    return env