# Setup
import argparse
import time
import torch
import torch.multiprocessing as _mp
#from test import test

# import the game
import gym_super_mario_bros
# import the joypad wrapper
from nes_py.wrappers import JoypadSpace
# import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT ## [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
# import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt
import numpy as np

#from train import train_model

# Collect arguments passed by meeee!
def parse():
    parser = argparse.ArgumentParser(description="CS696 RL PROJECT4")
    parser.add_argument('--world', default=1, help='Indicate World: [1,2,3,4,5,6,7,8]')
    parser.add_argument('--stage', default=1, help='Indicate Stage: [1,2,3,4]')
    parser.add_argument('--action_type', default='simple', help='Action Space: right/simple/complex')
    parser.add_argument('--train', action='store_true', help='Train a Model')
    parser.add_argument('--test', action='store_true', help='Test a Model')
    parser.add_argument('--record', action='store_true', help='Record video')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def run(args, record_video=False):
    start_time = time.time()
    if args.train:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)
        mp = _mp.get_context("spawn")


if __name__ == '__main__':
    args = parse()
    run(args, record_video=args.record)
