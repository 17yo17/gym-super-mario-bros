# import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import FrameStack, GrayScaleObservation
# import Vectorization Wrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

def processing_env(env):

    # 1. Grayscale  
    env = GrayScaleObservation(env, keep_dim=True) # Box(224, 240, 1): 224x240 pixels with 1 gray scaled channel

    # 2. Wrap inside the Dummy Environment
    env = DummyVecEnv([lambda: env])

    # 3. Stack the frames
    env = VecFrameStack(env, 4, channels_order='last')

    return env
