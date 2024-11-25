import numpy as np
import cv2
import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
# allows A3C to multiprocess
import torch.multiprocessing as mp
# Define action space
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
# Allow to reduce action space
from nes_py.wrappers import JoypadSpace
# Allows to spawn new process
import subprocess
# Directory to save 
SAVE_FILE = "save_trained_model/output.mp4"

class Monitor:
    def __init__(self, width, height):
        self.command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', 
                        '-s',  "{}X{}".format(width, height), '-r', '30', '-i', '-', '-an', '-vcodec', 'mpeg4', SAVE_FILE]

        try:
            self.pipe = subprocess.Popen(self.command, stdin=subprocess.PIPE)
            print("RECORDING VIDEO")
        except FileNotFoundError:
            print("FAILED TO RECORD VIDEO")
            pass
    
    def record_frame(self, image_array):
        if self.pipe:
            self.pipe.stdin.write(image_array.tobytes())

def process_frame(frame):
    if frame is not None:
        # RGB to Gray scale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize and Normalize
        frame = cv2.resize(frame, (84,84))[None,:,:] / 255.
        return frame # numpy array (1, 84, 84)
    else:
        return np.zeros((1,84,84))
        
class CustomReward(Wrapper):
    def __init__(self, env=None, world=None, stage=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage
        self.count_stacked = 0 # check if the agent is stucked

        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record_frame(state)

        state = process_frame(state)
        # Get reward gained from the previous state to current state and divide by 40
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            # Reached GOOoooaaaL!
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        # Count how long the agent is stacked
        if self.current_x == info["x_pos"]:
            self.count_stacked += 1
            if self.count_stacked >= 25:
                reward -= 10
        self.current_x = info["x_pos"]

        return state, reward/10., done, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        self.count_stacked = 0
        return process_frame(self.env.reset())
            
class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        # Accumulate all rewards (include rewards from skipped frames)
        total_reward = 0
        # Stack frames
        last_states = []
        # Start skipping frames
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        
        # Stack the last two frames (i = 2 and 3): (2,84,84) -> (84,84)
        # Maxpool to extract important info from a frame
        max_state = np.max(np.concatenate(last_states, 0), 0)
        # Shift one: first 3 frames will become next three frames
        self.states[:-1] = self.states[1:]
        # Append the maxpooled frame to at the end of the states stack
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32)

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return self.states.astype(np.float32)
        
def create_train_env(world, stage, action_type):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    
    # Monitor process
    monitor = Monitor(256, 240)

    # Define agent's action space
    if action_type == 'right':
        actions = RIGHT_ONLY
    elif action_type == 'simple':
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    # Reduce Action Space
    env = JoypadSpace(env, actions)
    # Customize Reward: [-15, 15] -> [CUSTOM]
    env = CustomReward(env, world, stage, monitor)
    # Stack 4 frames
    env = CustomSkipFrame(env)
    return env, env.observation_space.shape[0], len(actions)


# Test the environment
if __name__=="__main__":
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(1, 1))
    #env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v3".format(1, 1)) # for training
    # Monitor process
    monitor = Monitor(256, 240)

    # Reduce Action Space
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    print(env.action_space)

    # Customize Reward: [-15, 15] -> [-50, 50] 
    env = CustomReward(env, 1, 1, monitor)
    print(env.observation_space)

    # Stack 4 frames
    env = CustomSkipFrame(env)
    print(env.observation_space.shape)        

    
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        env.step(action)