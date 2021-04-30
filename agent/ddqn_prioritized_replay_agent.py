import numpy as np
import random
from collections import namedtuple, deque

from agent.dqn_model import QNetwork
from utils.utils import timed

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def annealing_generator(start: float,
                        end: float,
                        factor: float):
    decreasing = start > end
    
    eps = start
    while True:
        yield eps
        f = max if decreasing else min
        eps = f(end, factor*eps)
        
class Agent():
    '''
    DDQN Agent for solving the navegation system project.
    '''
    """Interacts with and learns from the environment."""

    def __init__(self, 
                 state_size, 
                 action_size,
                 hyperparams,
                 seed = 13):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.epsilon_gen = annealing_generator(start=hyperparams['eps_start'],
                                               end=hyperparams['eps_end'],
                                               factor=hyperparams['eps_decay'])

        self.beta_gen =    annealing_generator(start=hyperparams['beta_start'],
                                               end=hyperparams['beta_end'],
                                               factor=hyperparams['beta_factor'])
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, 
                                       action_size, 
                                       hyperparams['topology'],
                                       seed).to(device)
        self.qnetwork_target = QNetwork(state_size, 
                                        action_size, 
                                        hyperparams['topology'],
                                        seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, 
                                              BATCH_SIZE, 
                                              seed,
                                              per_epsilon = hyperparams.get('per_epsilon'),
                                              per_alpha = hyperparams.get('per_alpha'))

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        self.criterion = nn.MSELoss(reduce = False)
    
    @timed()
    def step(self, 
             state:torch.Tensor, 
             action:int, 
             reward:float, 
             next_state:torch.Tensor, 
             done:bool):
        '''
        Function to be called after every interaction between the agent
        and the environment.
        
        Updates the memory and learns.
        '''
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                self.learn(GAMMA)

    @timed()
    def act(self, 
            state: torch.Tensor, 
            training:bool = True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            training (bool): whether the agent is training or not.
        """
        eps = next(self.epsilon_gen)
        self.beta = next(self.beta_gen)
        rand = random.random()
        if training and rand < eps:
            # eps greedy exploration.
            return random.choice(np.arange(self.action_size))
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy action selection
            return np.argmax(action_values.cpu().data.numpy())

            

    @timed()
    def learn(self, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            gamma (float): discount factor
        """
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            gamma (float): discount factor
        """
        
        memory_indices,priorities,experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        
        self.optimizer.zero_grad()
        
        output = self.qnetwork_local.forward(states).gather(1,actions)
        
        # Build the targets
        
        self.qnetwork_local.eval() #Use the local network for calculating the indices of the max.
        with torch.no_grad():
            local_estimated_action_values = self.qnetwork_local(next_states)
        
        local_network_max_indices = torch.max(local_estimated_action_values,dim=1)[1].reshape(-1,1)
        
        self.qnetwork_target.eval() #Use the target network for using the estimated value.
        with torch.no_grad():
            target_estimated_action_values = self.qnetwork_target(next_states)
            
        
        estimated_max_value = target_estimated_action_values.gather(1,local_network_max_indices)
        
        labels = rewards+ (1-dones)*gamma*estimated_max_value
        
        self.memory.update_batches(memory_indices, (output-labels))
        
        beta =self.beta

        bias_correction = ((1/len(self.memory))*(1/priorities))**beta
        bias_correction = bias_correction/torch.max(bias_correction)
        # if random.random()>0.999:
        #     print(memory_indices,priorities,bias_correction)
            
        loss = (self.criterion(output, labels)*bias_correction).mean()
        
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  # TODO WHY?! Every learning we change the target?                
    @timed()
    def soft_update(self, 
                    local_model:nn.Module, 
                    target_model:nn.Module, 
                    tau:float):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        This is an alterative to the original formulation of the DQN 
        paper, in which the target agent is updated with the local 
        model every X steps.
        
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class SumTree(object):
    
    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        self.data_pointer = 0 # Pointer to the next leave to update.
        
        # Contains the experiences (so the size of data is capacity)
        self.data = [None]*capacity

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + (self.capacity - 1)

        """ tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update (tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0
            
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node 
    
    @property
    def maximum_priority(self):
        return np.max(self.tree[-self.capacity:]) # Returns the root node 

    def __len__(self):
        """Return the current size of internal memory."""
        return np.sum(~(self.tree[-self.capacity:]==0))

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples.
       Leverages a SumTree for efficiently sampling."""

    def __init__(self, 
                 buffer_size, 
                 batch_size, 
                 seed,
                 per_epsilon:float=None,
                 per_alpha:float=None,):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.per_epsilon = per_epsilon or 0.0001
        self.per_alpha = per_alpha or 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        maximum_priority = self.tree.maximum_priority +self.per_epsilon #TODO use clipped abs error?
        if maximum_priority==0:
            maximum_priority =1
        self.tree.add(maximum_priority,e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = []
        indices = []
        priorities = []
        # We divide the priority into buckets and sample from each of those
        segments = self.tree.total_priority/self.batch_size
        values = []
        for i in range(self.batch_size):
            value = random.uniform(i*segments, (i+1)*segments)
            leaf_index, priority, data = self.tree.get_leaf(value)
            
            experiences.append(data)
            indices.append(leaf_index)
            priorities.append(priority)
            values.append(value)

        # if random.random()>0.999:
        #     import pdb; pdb.set_trace()
                          
        try:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        except:
            import pdb; pdb.set_trace()
            
        return indices, torch.Tensor(priorities),(states, actions, rewards, next_states, dones)


    def update_batches(self,indices, errors):
        try:
            
            for index,error in zip(indices, errors.detach().numpy()):
                assert not np.isnan(error)
                self.tree.update(index,(abs(error)+self.per_epsilon)**self.per_alpha)
        except:
            import pdb; pdb.set_trace()            
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.tree)
    
