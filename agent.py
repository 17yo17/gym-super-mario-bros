import torch
# Allow Multiple Processes for A3C
import torch.multiprocessing as _mp
from env import create_train_env
from model import ActorCritic
from optimizer import GlobalAdam
from process import local_train
from process import local_test
# Define Hyperparameters
LR = 1e-4

class Agent(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_processes = args.num_processes
        
        # Check CUDA availability and set up SEED
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)
        # Creates a new Python interpreter for each process
        self.mp = _mp.get_context("spawn")
        self.env, self.num_states, self.num_actions = create_train_env(args.world, args.stage, args.action_type)
        # Global Model
        self.global_model = ActorCritic(self.num_states, self.num_actions).to(self.device)
        # Share parameters of global model in CPU memory across each process
        self.global_model.share_memory()

        self.optimizer = GlobalAdam(self.global_model.parameters(), lr=LR)
        

    def train(self):
        processes = []
        for index in range(self.num_processes):
            # Save only one of the process
            if index == 0:
                process = self.mp.Process(target=local_train, args=(index, self.args, self.global_model, self.optimizer, True))
            else:
                process = self.mp.Process(target=local_train, args=(index, self.args, self.global_model, self.optimizer))

            # Start Process and store it in the stack
            process.start()
            processes.append(process)
            for process in processes:
                # Wait for child processes complete its execution and aggregate the results
                # Prevent race condition
                process.join()

    def test(self):
        processes = []
        process = self.mp.Process(target=local_test, args=(self.num_processes, self.args, self.global_model))
        # Start Process and store it in the stack
        process.start()
        processes.append(process)
        for process in processes:
            process.join()
        
        