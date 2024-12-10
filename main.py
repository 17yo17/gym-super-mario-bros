# Setup
# argument parser
import argparse
# time watch
import time

import torch
import torch.multiprocessing as _mp

import gym_super_mario_bros
from agent import Agent


# Collect arguments passed by meeee!
def parse():
    parser = argparse.ArgumentParser(description="CS696 RL PROJECT4")
    parser.add_argument('--world', default=1, help='Indicate World: [1,2,3,4,5,6,7,8]')
    parser.add_argument('--stage', default=1, help='Indicate Stage: [1,2,3,4]')
    parser.add_argument('--action_type', default='simple', help='Action Space: right/simple/complex')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of environments to process simultaneously')
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
     
    agent = Agent(args)
    
    if args.train:
        agent.train()
    if args.test:
        agent.test()

if __name__ == '__main__':
    args = parse()
    run(args, record_video=args.record)
