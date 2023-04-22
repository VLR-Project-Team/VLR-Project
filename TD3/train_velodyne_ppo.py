import os
import time

# import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal

# from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
            nn.Linear(1024, 256),
            # nn.ReLU(),
        )
        # self.gru = nn.GRU(512, 256, 1, batch_first=True)
        self.fc = nn.Linear(256, action_dim)

        self.tanh = nn.Tanh()

    # def forward(self, s, h):
    def forward(self, s):
        # s = F.relu(self.layer_1(s))
        # s = F.relu(self.layer_2(s))
        # a = self.tanh(self.layer_3(s))
        print(s)
        s = self.mlp(s)
        s = s.unsqueeze(1)
        # s, h = self.gru(s, h)
        s = s.squeeze(1)
        a = self.tanh(self.fc(s))
        return a
        # return a, h


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

        return q

class PPO(object):
    def __init__(self, env, obs_dim, action_dim):
        self._init_hyperparameters()

        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor = Actor(self.obs_dim, self.action_dim)
        self.critic = Critic(self.obs_dim, self.action_dim)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode
        self.gamma = 0.99                          # discount factor
        self.n_updates_per_iteration = 5           # num of times to update actor/critic per iteration
        self.lr = 0.0003                           # learning rate of actor optimizer
        self.clip = 0.2                            # clip parameter for PPO

    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # Number of timesteps run so far this batch
        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float).to(device)
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        mean = self.actor(obs)

        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach().numpy()
    
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def learn(self, total_timesteps):
        t_so_far = 0

        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

            A_k = batch_rtgs - V.detach()

            # Normalizing advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = F.mse_loss(V, batch_rtgs)

                self.action_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.action_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            t_so_far += sum(batch_lens)

if __name__ == "__main__":
    seed = 0
    environment_dim = 20
    robot_dim = 4
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2

    env = GazeboEnv("multi_robot_scenario.launch", environment_dim)

    ppo = PPO(env, state_dim, action_dim)
    ppo.learn(1000000)
