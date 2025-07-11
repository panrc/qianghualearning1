import random
from collections import deque
import numpy as np
import torch

from . import config

class SumTree:
    """
    Sum tree data structure for efficient prioritized sampling
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.full = False

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
            self.full = True

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]
    
    @property
    def max_priority(self):
        return np.max(self.tree[-self.capacity:])
    
    @property
    def min_priority(self):
        return np.min(self.tree[-self.capacity:][self.tree[-self.capacity:] > 0])

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    """
    def __init__(self, capacity=config.REPLAY_MEMORY_SIZE, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta_start = beta_start  # Initial importance sampling weight
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 0.01  # Small amount to avoid zero priorities
        self.abs_error_upper = 1.0  # Clipped abs error

    def push(self, state, action, reward, next_state, done, error=None):
        """
        Store transition with priority based on TD error
        """
        if error is None:
            priority = self.tree.max_priority if self.tree.max_priority > 0 else self.abs_error_upper
        else:
            priority = min((abs(error) + self.epsilon) ** self.alpha, self.abs_error_upper)
        
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample batch with priorities and return importance sampling weights
        """
        batch_idx = []
        batch_data = []
        is_weights = []
        
        # Calculate current beta
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        
        priority_segment = self.tree.total_priority / batch_size
        min_prob = self.tree.min_priority / self.tree.total_priority if self.tree.min_priority > 0 else 1e-6
        
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            
            value = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(value)
            
            # Calculate importance sampling weight
            prob = priority / self.tree.total_priority
            is_weight = (prob / min_prob) ** (-beta)
            
            batch_idx.append(idx)
            batch_data.append(data)
            is_weights.append(is_weight)
        
        self.frame += 1
        
        # Unpack batch data
        states, actions, rewards, next_states, dones = zip(*batch_data)
        
        # Unpack states and next_states
        market_states, account_states = zip(*states)
        next_market_states, next_account_states = zip(*next_states)
        
        # Convert to tensors directly for better performance
        market_states = torch.FloatTensor(np.array(market_states)).to(config.DEVICE)
        account_states = torch.FloatTensor(np.array(account_states)).to(config.DEVICE)
        actions = torch.LongTensor(np.array(actions)).to(config.DEVICE)
        rewards = torch.FloatTensor(np.array(rewards)).to(config.DEVICE)
        next_market_states = torch.FloatTensor(np.array(next_market_states)).to(config.DEVICE)
        next_account_states = torch.FloatTensor(np.array(next_account_states)).to(config.DEVICE)
        dones = torch.BoolTensor(np.array(dones)).to(config.DEVICE)
        is_weights = torch.FloatTensor(is_weights).to(config.DEVICE)
        
        return (market_states, account_states, actions, rewards, 
                next_market_states, next_account_states, dones, 
                is_weights, batch_idx)

    def update_priorities(self, batch_idx, errors):
        """
        Update priorities based on TD errors
        """
        for idx, error in zip(batch_idx, errors):
            priority = min((abs(error) + self.epsilon) ** self.alpha, self.abs_error_upper)
            self.tree.update(idx, priority)

    def __len__(self):
        return min(self.tree.data_pointer if not self.tree.full else self.capacity, self.capacity)

# Legacy replay buffer for backward compatibility
class ReplayBuffer:
    def __init__(self, capacity=config.REPLAY_MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Saves a transition.
        The state and next_state are tuples: (market_data, account_data)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a batch of transitions.
        """
        # The states, actions, etc. for the whole batch
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))

        # Unpack the states and next_states into their market and account components
        market_states, account_states = zip(*states)
        next_market_states, next_account_states = zip(*next_states)

        # Convert to tensors directly for better performance
        market_states = torch.FloatTensor(np.array(market_states)).to(config.DEVICE)
        account_states = torch.FloatTensor(np.array(account_states)).to(config.DEVICE)
        actions = torch.LongTensor(np.array(actions)).to(config.DEVICE)
        rewards = torch.FloatTensor(np.array(rewards)).to(config.DEVICE)
        next_market_states = torch.FloatTensor(np.array(next_market_states)).to(config.DEVICE)
        next_account_states = torch.FloatTensor(np.array(next_account_states)).to(config.DEVICE)
        dones = torch.BoolTensor(np.array(dones)).to(config.DEVICE)

        return market_states, account_states, actions, rewards, next_market_states, next_account_states, dones

    def __len__(self):
        return len(self.buffer) 