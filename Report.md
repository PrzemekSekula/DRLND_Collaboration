[//]: # (Image References)

[image1]: ./img/averaged_scores_train.jpg "Averaged training scores"
[image2]: ./img/averaged_scores_train3500.jpg "Averaged training scores - from episode 3500"
[image3]: ./img/MADDPG.jpg "MADDPG - Overview"

# Collaboration and Competition - Report

## Overview

- The goal of this project was to train an agent (or agents) that controls a tennis player.
- I used slightly modified version of MADDPG algorithm. The main modification is that the collecting episodes and training the agents are being done asynchronically. Additinally I used uniformly random noise instead of Ornstein-Uhlenbeck noise.
In order to solve this task I modified my code form the previous Udacity RL Nanodegree assignment [Continuus Control](https://github.com/PrzemekSekula/DRLND_ContinuousControl.git).
- It took 3,440 episodes to reach the goal for the first time (100 episode window average score >= 0.5), and 3,759 episodes to reach a stable solution. 
- I trained the agent until it achieves 2.0 score.
- This code uses a Unity environment provided by Udacity. For more information check the `README.md` where the installation process is described.


## Learning Algorithm

From the implementation point of view, the MADDPG algorithm is just a modification of DDPG. The easiest way to explain the algorithm is to do it incrementally. My explanation follows this logic. The algorithm's description looks as follows:
- **DDPG**: description of the original algorithm that underlays the MADDPG 
- **MADDPG**: Description of MADDPG as a modification of the DDPG algorithm
- **One agent**: Explanation - why did I use one agent instead of two.
- **Parallelized solution**: Implemented, parallel solution.


#### DDPG (base algorithm)

DDPG is a modification of DQN algorithm destined for solving continuous control tasks. The algorithm was introduced by T. Lillicrap et. al. in 2015 and it is described [here](https://arxiv.org/abs/1509.02971).

In DDQN we have two networks.
- The Actor network that is responsible for maping the policy function $State\rightarrow Action$
- The Critic network that is responsible for estimating State-action values $Q(S, a)$

Both networks have two copies, (local and target). The local networks are updated using the
loss functions, whereas the target networks are updated using the soft update rule:
$target weights = (1-\tau)*target weights + \tau * local weights$


The original DDQN algorithm looks as follows:

- Initialize Critic and Actor Networks
- Initialize local copies of Critic and Actor Networks
- Initialize replay buffer
- Repeat for N episodes:
    - receive the initial state
    - repeat for t steps (or until the episode is finished):
        - select an action acording to the current policy (with noise)
        - execute ation, observe reward and next state
        - store (state, action, reward, next_state) in replay buffer
        - sample a minibatch from the replay buffer
        - update (local) critic and actor networks
        - update target networks using soft updates.


**Noise**

In order to ensure exploration, the [Ornstein-Uhlenbeck process](https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823) noise is added to each action during training. The Ornstein-Uhlenbeck process models the velocity of a Brownian particle with friction, which results in temporally correlated values centered around 0.

#### MADDPG algorithm
MADDPG stands for DDPG. It has been proposed by Ryan Love, Yi Wu et al. in 2017 in the [Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) paper.
The algorithm is a DDPG modification designed for solving multi-agent environments. In theory, it is possible to and independent DDPG for each agent. However, in classic DDPG, every other agent would be considered only as a part of a non-stationary environment. Additionally, each agent has access only to its own observations and cannot learn based on other agents' observations. This approach can make the learning process slow and inefficient.
In MADDPG the critic network has access not only to its own agent but also to the observations and actions of all other agents. It results in faster, and more stable learning.

The best explanation of MADDPG is the following image (from MADDPG paper)
![MADDPG][image3]


#### One agent
In theory, MADDPG is designed to work with more than one agent. Each agent should have its own critic and actor networks. Although, in this task, the environment is symmetrical. Both agents get the same information (respective to their location). It means that the agent trained to play on the left side of the field can be used to play on the right side on the field. It implies that it is enough to create one agent and let it control both players.

In my code, I used one agent to control both players. However, I wanted to make my code more general, so everything can work with two agents as well. It has some disadvantages, namely:
- The replay buffer is slower. Had it been designed for one agent only, it would have worked faster.
- In every training step, the same agent is trained twice, from the perspective of the left and the right player.
Despite these disadvantages, the code behaved quite well, and it will be easier to generalize it in the future.


#### Parallelized MADDPG (implemented solution)
Although MADDPG and DDPG seem to be very clear, there is a drawback in them. Namely, they are not parallelized. Between each two learning steps, agents have to wait for one episode step, and - while the agent is learning - the episodes are not generated. As the agents use replay buffer and MADDPG is an off-policy algorithm, there is no need to wait for a response from the environment before the next training steps. I simply parallelized these to steps, so my algorithm looks as follows:

*Initialization*
- Initialize Critic and Actor Networks (for each agent)
- Initialize local copies of Critic and Actor Networks (for each agent)
- Initialize replay buffer

*Thread 1 - Interacting with environment*
- Repeat for N episodes
    - receive the initial state
    - repeat for t steps (or until the episode is finished):
        - select the actions acording to the current policies (with noise)
        - execute ations, observe rewards and next states
        - store (states, actions, rewards, next_states) in the Replay Buffer

*Thread 2 - Learning*
- Repeat as long as episodes are generated
    - sample a minibatch from the replay buffer
    - update (local) critic and actor networks for each agent
    - update target networks using soft updates for each agent.

This modification eliminates the delays that occur due to the sequential implementation of the algorithm.

**Noise**

In my algorithm I changed the OU noise to uncorrelated uniformly random distribution and it speeds up the learning process. The noise was computed according to the formula:

<code>
    sigma * np.random.rand(len(action)) - sigma / 2
</code>

where the coefficient $\sigma$ (sigma) is the range of the noise.

Additionally the noise has been reduced as the agent performed better.
There is no deep explanation for such changes. I experimented a lot with DDPG noise to choose the best one. Here I just took the DDPG solution and it worked, so I didn't change it.


#### Neural Networks
**Critic**
A fully connected (dense) neural network were used, as critic network.
The input layer are just concatenated states and values for both agents (or, in this case, for one agent but from two points of view)
A fully  of the `Deep and Wide` network was used as a critic network. The parameters are as follows:
- Input layer (1): 2 * (State vector + action vector), 52 dimensions
- First layer: 256 neurons
- Second layer: 128 neurons. 
- Third layer: 64 neurons
- Output layer: 1 neuron

The output layer uses linear activation function, all other activation functions are ReLU.

**Actor**
A fully connected (dense) neural network were used, as an actor network.
The paraemeters of the network where as follows:
- Input layer (1): State vector, 33 dimensions
- First layer: 256 neurons, ReLU activation function.
- Second layer: 128 neurons, ReLU activation function.
- Third layer: 64 neurons, ReLU activation function.
- Output layer - 4 neurons, Tanh activation function. (4 neurons correspond to 4 dimensions of the action vector).

The $tanh$ activation function at the output layer is used for clipping. In this environment, all the actions shall be scaled from -1 to 1, and this is what $tanh$ is doing.

#### Hyperparameters
As a base I took  the hyperparameters from my DDPG solution. The parametrs (kept in the `Params` class) are as follows:

<code>
BUFFER_SIZE = int(5e5)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
</code>

I also used the noise-parameters that are defined in the `generate_episode` function. They are as follows:
<code>
n_episodes=2000       # Number of episodes
noise_max = 0.1       # maximum sigma for noise generation
noise_min = 0.05      # minimum sigma for noise generation
min_noise_level = 1.5 # averaged score that corresponds to the minimum sigma for noise generation
</code>

## Implementation

#### Code decsription
The [Udacity DDPG](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) implementation was used as a base code, with only minor changes. The structure of the files is as follows:

- `models.py` - contains the actor and critic pytorch models
- `ddpg_agent.py` - contains the gist of the algorithm namely:
    - `OUNoise` class is responsible for generating noise that is added to actions to ensure exporation. As this noise is temporarily correlated, it is natural to implement it in a separate class.
    - `ReplayBuffer` class that can store and sample experience tupples
    - `Agent` class - the main class with the following methods:
        - `act` - the method that takes states as input and returns the corresponding action, according to the current policy. 
        - `reset` - resets noise parameters (useful with OU noise, not used in my solution)
        - `learn` - performs actor and critic learning for one batch
        - `soft_update` - updates target networks (called by `learn`)
- `Tennis.ipynb` - notebook with training code. Points 1-3 of this notebook are loading and presenting the environment. The task starts in point 4:
    - `train_agent` trains an agent. To start training the Replay Buffer must be largest than the batch size. Training is performed as long as the global variable `run_training == True`
    - `generate_episodes` generates episodes. To keep everything as similar to Udacity example as possible, this method is also responsible for collecting printing the on-going training results and saving the models.  
- `AnalyzeResults.ipynb` - notebook that analyzes the training and testing results.
- `models/checkpoint_actor.pth` - final actor network 
- `models/checkpoint_critic.pth` - final critic network 

#### Experiment description
Only the final (working) experiment is described here.
The Agent was trained untill at least one of the following conditions is fulfilled:
- 10,000 episodes were generated.
- Average score is at least 2.0.
- Training took more than 1 hour.



3,837 episodes were generated, which corresponds to 211,085 steps. In the meantime 22,134 learning steps has been performed which gives 9.54 episode steps / learning step. During each learning step, the agent was trained twice on the 512-element batch (from left and right player perspective).

The training scores averaged with moving average (window size = 100 episodes) are presented in the figure below. 
![Scores][image1]
All the metrics in the chart are computed for the last 100 episodes.
- Red line stands for the mean score.
- Blue area stands for the standard deviation of the scores.

Thus the performance improved mostly in the last 1,000 episodes, a cropped version of the figure is also presented.
![Scores][image2]


It took 3,440 episodes to reach the goal for the first time (100 episode window average score >= 0.5), and 3,759 episodes to reach a stable solution. 

## Colclusions
I think it was a lucky strike to start with modifying my DDPG code. The tuning of the algorithm was very easy, in fact, I did not change the parameters that worked for me in the Continuous Control assignment. I should probably try a more demanding task, it is difficult to make some conclusions if everything went well so smoothly. 

## Ideas for future work
- Learn how to create/modify Unity environments. 
- I should reconsider the Replay Buffer. Originally I was thinking about using Prioritized Experience Replay, but I wasn't sure, how to do it in case of more than agents (each agent could have different priorities). I have a few ideas but I need to test them.
- The Replay Buffer should also be parallelized (next batch should be prepared while the current batch is used for training). In my implementation of DDPG, the batch generation was responsible for 12% of the learning time. Here the Replay buffer is even more complicated, so it was probably even more computational-expensive.

