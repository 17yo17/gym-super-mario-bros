import timeit
import torch
from torch.utils.tensorboard import SummaryWriter

from env import create_train_env
from model import ActorCritic

LOG_PATH = "logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = 500


def local_train(index, args, global_model, optimizer, save=False):
    # Check CUDA availability and set up SEED
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(LOG_PATH)

    env, num_states, num_actions = create_train_env(args.world, args.stage, args.action_type)

    # Model for Child Process
    local_model = ActorCritic(num_states, num_actions).to(DEVICE)
    # Set the agent to Training mode
    local_model.train()

    state = torch.from_numpy(env.reset()).to(DEVICE)

    done = True
    curr_step = 0
    curr_episode = 0

    while True:
        if save:
            if curr_episode % CHECKPOINT == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(), "checkpoint_{}".format(curr_episode))
            print("Process {}: Episode {}".format(index, curr_episode))

        curr_episode += 1
        # Child Process gets the current parameters of the Global Model
        local_model.load_state_dict(global_model.state_dict())

        
        break
    
    


    
def local_test(index, args, global_model):
    pass