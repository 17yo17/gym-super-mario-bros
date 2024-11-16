# Setup
# import the game
import gym_super_mario_bros
# import the joypad wrapper
from nes_py.wrappers import JoypadSpace
# import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT ## [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]

env = gym_super_mario_bros.make("SuperMarioBros-v0")
#print(env.action_space) Discrete(256): 256 possible actions
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


