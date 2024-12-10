import torch
import torch.nn.functional as F
# Allow Multiple Processes for A3C
import torch.multiprocessing as _mp
from env import create_train_env
from model import ActorCritic
from optimizer import GlobalAdam
from process import local_train
from process import local_test

# Load the checkpoint
try:
    PATH = 'my_trained_models/a3c_super_mario_bros_1_1_10000'
    checkpoint = torch.load(PATH, weights_only=True)
    print(checkpoint.keys())
    model_weights = checkpoint['model_state_dict']
    print("Check Point is loaded!")
except:
    print("Check Point is not loaded!")
    pass

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

        if args.train:
            # Creates a new Python interpreter for each process
            self.mp = _mp.get_context("spawn")
            self.env, self.num_states, self.num_actions = create_train_env(args.world, args.stage, args.action_type, args.record)
        
            # Global Model
            self.global_model = ActorCritic(self.num_states, self.num_actions).to(self.device)
            # Share parameters of global model in CPU memory across each process
            self.global_model.share_memory()

            self.optimizer = GlobalAdam(self.global_model.parameters(), lr=LR)
            
        elif args.test:
            self.env, self.num_states, self.num_actions = create_train_env(args.world, args.stage, args.action_type, args.record)
            self.model = ActorCritic(self.num_states, self.num_actions)
            self.model.load_state_dict(torch.load("my_trained_models/a3c_super_mario_bros_1_1_10000"))
            self.model.to(self.device)
            self.model.eval()

        else:
            print("Choose Train/Test")
        

    def train(self):
        processes = []
        for index in range(self.num_processes):
            # Managing Child Processes
            if index == 0:
                process = self.mp.Process(target=local_train, args=(index, self.args, self.global_model, self.optimizer, True))
            else:
                process = self.mp.Process(target=local_train, args=(index, self.args, self.global_model, self.optimizer))
            process.start()
            processes.append(process)
                
        # Evaluate Global Model's Performance
        #process = self.mp.Process(target=local_test, args=(self.num_processes, self.args, self.global_model))
        #process.start()
        #processes.append(process)
        
        for process in processes:
            process.join()

        
    def test(self):
        state = torch.from_numpy(self.env.reset())
        done = True

        while True:
    
            if done:
                h_0 = torch.zeros((1,512), dtype=torch.float)
                c_0 = torch.zeros((1,512), dtype=torch.float)
                self.env.reset()
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

            state = state.to(self.device)
            h_0 = h_0.to(self.device)
            c_0 = c_0.to(self.device)
            
            logits, value, h_0, c_0 = self.model(state, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()
            action = int(action)
            state, reward, done, info = self.env.step(action)
            state = torch.from_numpy(state)
            #self.env.render()
            
            if info["flag_get"]:
                print("World {} stage {} completed".format(self.args.world, self.args.stage))
                break
            
                




