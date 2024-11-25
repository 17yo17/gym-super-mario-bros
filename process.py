import timeit
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from env import create_train_env
from model import ActorCritic

LOG_PATH = "logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LOCAL_STEPS = 50
NUM_EPISODES = 5e6
CHECKPOINT = int(NUM_EPISODES / NUM_LOCAL_STEPS)

# Hyperparameters
GAMMA = 0.9
TAU = 1.0
BETA = 0.01

def save_checkpoint(model, optimizer, episode, filename="checkpoint.pth"):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at episode {episode}")

def local_train(index, args, global_model, optimizer, save=False):
    # Check CUDA availability and set up SEED
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(LOG_PATH)

    env, num_states, num_actions = create_train_env(args.world, args.stage, args.action_type, args.record)

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
            h_0 = h_0.detach().to(DEVICE)
            c_0 = c_0.detach().to(DEVICE)

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

            # Sample based on the probability
            m = Categorical(policy)
            action = m.sample().item()

            state, reward, done, info = env.step(action)
            state = torch.from_numpy(state).to(DEVICE)

            if curr_step > NUM_EPISODES:
                done = True

            # Reached terminate state, reset
            if done:
                curr_step = 0
                state = torch.from_numpy(env.reset()).to(DEVICE)

            # Save Logs
            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done: 
                break

        # Bootstrap: estimate of the future reward if not terminated (Critic output)
        R = torch.zeros((1,1), dtype=torch.float).to(DEVICE)
        if not done: 
            _, R, _, _ = local_model(state, h_0, c_0)

        # Generalized Advantage Estimation
        gae = torch.zeros((1,1), dtype=float).to(DEVICE)
        
        # Loss Terms
        actor_loss = 0 # proportional to advantage A(s,a)
        critic_loss = 0 # squared error between R and critic's predicted value
        entropy_loss = 0 # penalizing determiinistic policy to explore
        next_value = R # the value of next state in the sequence for GAE claculations

        # Backpropagation Through Time with TD
        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:

            # GAMMA (degree of future rewards' contribution), TAU (smoothing factor to control the contribution of temporal difference)
            gae = gae * GAMMA * TAU
            # TD Residual
            gae = gae + reward + GAMMA * next_value.detach() - value.detach()
            
            next_value = value

            actor_loss = actor_loss + log_policy * gae # Maximize -> Increase prob of actions that leads to + advantages
            R = R * GAMMA + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2 # Minimize the squared error between the predicted value and bootstrap
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - BETA * entropy_loss
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward()

        # Synchronizing Gradients of Child Processes to Global Model
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param.grad = local_param.grad

        optimizer.step()

        # Save Checkpoint
        if curr_episode == CHECKPOINT:
            print("Training process {} terminated".format(index))
            if save:
                save_checkpoint(global_model, optimizer, curr_episode, filename="checkpoint_{}.pth".format(curr_episode))
                end_time = timeit.default.timer()
                print("The code run for %.2f s".format(end_time - state_time))
            return
    

    
def local_test(index, args, global_model):
    pass