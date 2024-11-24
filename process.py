import timeit
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from env import create_train_env
from model import ActorCritic

LOG_PATH = "logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = 500
NUM_LOCAL_STEPS = 50


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

        if done:
            # Initialize h0 and c0 all zeros
            h_0 = torch.zeros((1, 512), dtype=torch.float).to(DEVICE)
            c_0 = torch.zeros((1, 512), dtype=torch.float).to(DEVICE)
        else:
            # Detach the hidden state of LSTM
            h_0 = torch.detach().to(DEVICE)
            c_0 = torch.detach().to(DEVICE)

        log_policies = []
        values = []
        rewards = []
        entropies = []

        for _ in range(NUM_LOCAL_STEPS):
            curr_step += 1
            # Actor output, Critic output, Hidden and Cell states of the recurrent layer
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)

            # Convert Actor output to Density
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            # High: sparse probability (unsure about the action), Low: concentrated probability (confident about the action)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

                        
        break
    
    


    
def local_test(index, args, global_model):
    pass