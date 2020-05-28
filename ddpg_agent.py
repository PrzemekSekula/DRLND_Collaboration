import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, params, device = DEVICE, critic_input_size = None):
        """Initialize an Agent object.
        """
        
        self.params = params
        self.state_size = params.STATE_SIZE
        self.action_size = params.ACTION_SIZE
        self.seed = params.SEED
        self.tau = params.TAU
        
        self.device = device
        
        if critic_input_size is None:
            critic_input_size = 2 * (self.state_size + self.action_size)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), 
                                           lr=params.LR_ACTOR, weight_decay=params.WEIGHT_DECAY_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(critic_input_size, self.seed).to(device)
        self.critic_target = Critic(critic_input_size, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                           lr=params.LR_CRITIC, weight_decay=params.WEIGHT_DECAY_CRITIC)

        # Noise process
        self.noise = OUNoise(self.action_size, self.seed, 
                             mu=0., theta=params.NOISE_THETA, sigma=params.NOISE_SIGMA)
        
        # Parameters for learning
        self.gamma = params.GAMMA
        self.learning_step = 0 # Counter for learning steps
    
    def act(self, state, add_noise=False, sigma = 0.1):
        """
        Returns actions for given state as per current policy.
        Arguments:
            state - input state
            add_noise - can be:
                False   - No nose added (default)
                'OU'    - Ornstein-Uhlenbeck noise added
                'rand'  - uniformly random noise added
                'sigma' - noise is scaled from -simga/2 to sigma/2. Works with 'rand' noise
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            if add_noise == 'OU':
                action += self.noise.sample()
            else:
                action += sigma * np.random.rand(len(action)) - sigma / 2
                
            return np.clip(action, -1, 1) # Clipping is necessary if we are adding noise
        else:
            return action
        
    def reset(self):
        self.noise.reset()

    def learn(self, 
              states, actions, rewards, next_states, dones,
              next_actions, 
              ag2_states, ag2_actions, ag2_next_states, 
              ag2_next_actions):              
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            states, actions, rewards, next_states, dones - parameters for agent
            next_actions - actions produced by target network
            ag2_states, ag2_actions, ag2_next_states - parameters for the other agent
            ag2_next_actions - actions produced by target network of the other agent
        """

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, next_actions, ag2_next_states, ag2_next_actions)
        
            # Compute Q targets for current states 
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
            
        # Compute critic loss
        Q_expected = self.critic_local(states, actions, ag2_states, ag2_actions)
            
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        pred_actions = self.actor_local(states)
        actor_loss = -self.critic_local(states, pred_actions, ag2_states, ag2_next_actions).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, num_agents, buffer_size, seed, device=DEVICE):
        """Initialize a ReplayBuffer object.
        Params
        ======
            num_agents (int): number of agents that are acting in every step
            buffer_size (int): maximum size of buffer 
            seed (int): random seed
        """
        self.num_agents = num_agents
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        
        ret = [] #
        
        for agent_number in range(self.num_agents):
            states = torch.from_numpy(
                np.vstack([e.states[agent_number] for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(
                np.vstack([e.actions[agent_number]  for e in experiences if e is not None])).float().to(self.device)
            rewards = torch.from_numpy(
                np.vstack([e.rewards[agent_number]  for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(
                np.vstack([e.next_states[agent_number] for e in experiences if e is not None])).float().to(self.device)
            dones = torch.from_numpy(
                np.vstack([e.dones[agent_number] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

            ret.append((states, actions, rewards, next_states, dones))
        
        return ret
       
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)