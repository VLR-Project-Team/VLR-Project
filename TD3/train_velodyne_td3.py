import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # self.layer_1 = nn.Linear(state_dim, 800)
        # self.layer_2 = nn.Linear(800, 600)
        # self.layer_3 = nn.Linear(600, action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.ReLU(),
        )
        self.gru = nn.GRU(512, 256, 1, batch_first=True)
        self.fc = nn.Linear(256, action_dim)

        self.tanh = nn.Tanh()

    def forward(self, s, h):
        # s = F.relu(self.layer_1(s))
        # s = F.relu(self.layer_2(s))
        # a = self.tanh(self.layer_3(s))
        s = self.mlp(s)
        s = s.unsqueeze(1)
        s, h = self.gru(s, h)
        s = s.squeeze(1)
        a = self.tanh(self.fc(s))
        return a, h


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q = self.layer_3(s1)

        return 1

class PPO(object):
    def __init__(self, env, environment_dim, robot_dim, action_dim, total_timesteps):
        self.env = env
        self.state_dim = environment_dim * 10 + robot_dim
        self.action_dim = action_dim
        self.total_timesteps = total_timesteps
    
    def learn(self):
        timesteps = 0
        while timesteps < self.total_timesteps:
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                state, reward, done, _ = self.env.step(action)
                timesteps += 1

# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
eval_freq = 5e3  # After how many steps to perform the evaluation
max_ep = 500  # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 5e6  # Maximum number of steps to perform
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = (
    500000  # Number of steps over which the initial exploration noise will decay over
)
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 40  # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates
buffer_size = 1e6  # Maximum size of the buffer
file_name = "TD3_velodyne"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
load_model = False  # Weather to load a stored model
random_near_obstacle = True  # To take random actions near obstacles or not

# Create the network storage folders
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Create the training environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

# Create the network
network = TD3(state_dim, action_dim, max_action)
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

print("Model loaded!")

# Create evaluation data store
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# Begin the training loop
while timestep < max_timesteps:
    print("Timestep:", timestep)

    # On termination of episode
    if done:
        if timestep != 0:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        state = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # add some exploration noise
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    # If the robot is facing an obstacle, randomly force it to take a consistent random action.
    # This is done to increase exploration in situations near obstacles.
    # Training can also be performed without it
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # Save the tuple in replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Update the counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./models")
np.save("./results/%s" % file_name, evaluations)
