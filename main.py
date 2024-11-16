# Setup
# import the game
import gym_super_mario_bros
# import the joypad wrapper
from nes_py.wrappers import JoypadSpace
# import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT ## [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
# import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# import Vectorization Wrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

from train import train_model
from env_preprocessor import processing_env

# 1. Create the base environment
env = gym_super_mario_bros.make("SuperMarioBros-v0")
#print(env.action_space) Discrete(256): 256 possible actions

# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
##print(env.action_space) Discrete(7): 7 possible actions
#print(env.observation_space.shape) # Box(224, 240, 3): 224x240 pixels with 3 color channels

#########  TESTING  #########
# Create a flag - restart or not
#done = True
# Loop through each frame in the game
#for step in range(5000):
    # Start the game to begin with
#    if done:
        # Start the game
#        state = env.reset()
    # Get the next state, reward, done, and info based on the random action
#    state, reward, done, info = env.step(env.action_space.sample())
    # Display the game on the screen
#    env.render()
# Close the game
#env.close()
###########################

# 3. Preproces the environment
env = processing_env(env)

# 4. Train model
#train_model(env)

# 5. Test model
from test import test
#test_model(env)
from stable_baselines3 import PPO
env = gym_super_mario_bros.make("SuperMarioBros-v0")
model = PPO.load('./train/best_model_1000000')
test(model, env, total_episodes=100, record_video=True)