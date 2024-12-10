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
SAVE_FILE = "recorded_video/output.mp4"

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

    def render(self, frame):
        """Display the frame using OpenCV."""
        if frame is not None:
            # Ensure the frame is in RGB format
            if frame.shape[-1] != 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert if BGR
            cv2.imshow("Super Mario Bros", frame)

            # Wait for a short time and close window if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return False
        return True

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

        return state, reward/10., done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())
            
class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)
        
def create_train_env(world, stage, action_type, record=False):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    
    # Monitor process
    if record:
        monitor = Monitor(256, 240)
    else:
        monitor = None

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
    #env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(1, 1))
    #env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v3".format(1, 1)) # for training
    # Monitor process
    #monitor = Monitor(256, 240)

    # Reduce Action Space
    #env = JoypadSpace(env, SIMPLE_MOVEMENT)
    #print(env.action_space)

    # Customize Reward: [-15, 15] -> [-50, 50] 
    #env = CustomReward(env, 1, 1, monitor)
    #print(env.observation_space)

    # Stack 4 frames
    #env = CustomSkipFrame(env)
    #print(env.observation_space.shape)        
    env, num_states, num_actions = create_train_env(1, 1, 'simple', True)

    env.reset()
    for _ in range(2000):
        action = env.action_space.sample()
        s, r, d, i = env.step(action)
        if d:
            env.reset()