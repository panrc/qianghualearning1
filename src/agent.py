import math
import random
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from . import config
from .model import DoubleDuelingDQN_LSTM
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

class Agent:
    def __init__(self, use_prioritized_replay=True, use_noisy_net=False):
        self.device = torch.device(config.DEVICE)
        self.use_prioritized_replay = use_prioritized_replay
        self.use_noisy_net = use_noisy_net
        
        # Initialize networks
        self.policy_net = DoubleDuelingDQN_LSTM(use_noisy=use_noisy_net).to(self.device)
        self.target_net = DoubleDuelingDQN_LSTM(use_noisy=use_noisy_net).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # Choose replay buffer type
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(config.REPLAY_MEMORY_SIZE)
        else:
            self.memory = ReplayBuffer(config.REPLAY_MEMORY_SIZE)
        
        # LSTM state persistence
        self.policy_hidden_state = None
        self.target_hidden_state = None
        
        # Dynamic epsilon for exploration
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_MIN
        
        # Training statistics
        self.step_count = 0

    def reset_hidden_states(self):
        """Reset LSTM hidden states at the beginning of each episode"""
        self.policy_hidden_state = None
        self.target_hidden_state = None

    def update_epsilon(self):
        """Update epsilon for exploration"""
        if not self.use_noisy_net:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def select_action(self, state, epsilon=None):
        """Selects an action using epsilon-greedy policy or noisy networks"""
        if epsilon is None:
            epsilon = self.epsilon if not self.use_noisy_net else 0.0
            
        if self.use_noisy_net:
            # Noisy networks handle exploration automatically
            with torch.no_grad():
                market_state_t, account_state_t = self._prepare_tensors(state)
                q_values, self.policy_hidden_state = self.policy_net(
                    market_state_t, account_state_t, self.policy_hidden_state
                )
                return q_values.max(1)[1].view(1, 1).item()
        else:
            # Traditional epsilon-greedy
            if random.random() > epsilon:
                with torch.no_grad():
                    market_state_t, account_state_t = self._prepare_tensors(state)
                    q_values, self.policy_hidden_state = self.policy_net(
                        market_state_t, account_state_t, self.policy_hidden_state
                    )
                    return q_values.max(1)[1].view(1, 1).item()
            else:
                return random.randrange(config.NUM_ACTIONS)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        if self.use_prioritized_replay:
            # For prioritized replay, we need to calculate initial TD error
            with torch.no_grad():
                market_state_t, account_state_t = self._prepare_tensors(state)
                next_market_state_t, next_account_state_t = self._prepare_tensors(next_state)
                
                # 修复：正确获取 Q 值
                q_values, _ = self.policy_net(market_state_t, account_state_t)
                current_q = q_values[0, action]  # 使用 [0, action] 来正确索引批次中的第一个样本的特定动作
                
                if not done:
                    # Double DQN: use policy network to select action, target network to evaluate
                    next_q_policy, _ = self.policy_net(next_market_state_t, next_account_state_t)
                    next_action = next_q_policy.max(1)[1].item()
                    next_q_target, _ = self.target_net(next_market_state_t, next_account_state_t)
                    next_q = next_q_target[0, next_action]  # 同样修复索引方式
                    target_q = reward + config.GAMMA * next_q
                else:
                    target_q = reward
                
                td_error = abs(current_q.item() - target_q.item())
                self.memory.push(state, action, reward, next_state, done, td_error)
        else:
            self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        """Optimize the model using either prioritized or standard replay"""
        if len(self.memory) < config.BATCH_SIZE:
            return None

        if self.use_prioritized_replay:
            return self._optimize_with_prioritized_replay()
        else:
            return self._optimize_with_standard_replay()

    def _optimize_with_prioritized_replay(self):
        """Optimization with prioritized experience replay"""
        batch_data = self.memory.sample(config.BATCH_SIZE)
        (market_states_t, account_states_t, actions_t, rewards_t, 
         next_market_states_t, next_account_states_t, dones_t, 
         is_weights, batch_idx) = batch_data

        # Reset noise for noisy networks
        if self.use_noisy_net:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

        # Current Q values
        q_values, _ = self.policy_net(market_states_t, account_states_t)
        state_action_values = q_values.gather(1, actions_t.unsqueeze(1))

        # Next state Q values using Double DQN
        next_state_values = torch.zeros(config.BATCH_SIZE, device=self.device)
        non_final_mask = ~dones_t
        
        if non_final_mask.sum() > 0:
            # Double DQN: policy network selects actions, target network evaluates
            with torch.no_grad():
                non_final_next_market = next_market_states_t[non_final_mask]
                non_final_next_account = next_account_states_t[non_final_mask]
                
                # Policy network selects best actions
                next_q_policy, _ = self.policy_net(non_final_next_market, non_final_next_account)
                next_actions = next_q_policy.max(1)[1]
                
                # Target network evaluates selected actions
                next_q_target, _ = self.target_net(non_final_next_market, non_final_next_account)
                next_state_values[non_final_mask] = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # Compute expected Q values
        expected_state_action_values = rewards_t + (config.GAMMA * next_state_values)

        # Compute TD errors
        td_errors = state_action_values.squeeze(1) - expected_state_action_values
        
        # Weighted loss for prioritized replay
        loss = (is_weights * F.smooth_l1_loss(
            state_action_values.squeeze(1), expected_state_action_values, reduction='none'
        )).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in replay buffer
        td_errors_abs = td_errors.abs().detach().cpu().numpy()
        self.memory.update_priorities(batch_idx, td_errors_abs)

        self.step_count += 1
        return loss.item()

    def _optimize_with_standard_replay(self):
        """Standard DQN optimization"""
        market_states_t, account_states_t, actions_t, rewards_t, next_market_states_t, next_account_states_t, dones_t = \
            self.memory.sample(config.BATCH_SIZE)

        # Reset noise for noisy networks
        if self.use_noisy_net:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

        # Current Q values
        q_values, _ = self.policy_net(market_states_t, account_states_t)
        state_action_values = q_values.gather(1, actions_t.unsqueeze(1))

        # Next state Q values using Double DQN
        next_state_values = torch.zeros(config.BATCH_SIZE, device=self.device)
        non_final_mask = ~dones_t
        
        if non_final_mask.sum() > 0:
            with torch.no_grad():
                non_final_next_market = next_market_states_t[non_final_mask]
                non_final_next_account = next_account_states_t[non_final_mask]
                
                # Double DQN
                next_q_policy, _ = self.policy_net(non_final_next_market, non_final_next_account)
                next_actions = next_q_policy.max(1)[1]
                
                next_q_target, _ = self.target_net(non_final_next_market, non_final_next_account)
                next_state_values[non_final_mask] = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # Compute expected Q values
        expected_state_action_values = rewards_t + (config.GAMMA * next_state_values)

        # Compute loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        return loss.item()

    def update_target_net(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def _prepare_tensors(self, state):
        """Prepare state tensors for network input"""
        market_state, account_state = state
        
        # Convert market state to tensor and add batch dimension if needed
        if isinstance(market_state, np.ndarray):
            market_state = torch.FloatTensor(market_state)
        if market_state.dim() == 2:
            market_state = market_state.unsqueeze(0)
        
        # Convert account state to tensor and add batch dimension if needed
        if isinstance(account_state, np.ndarray):
            account_state = torch.FloatTensor(account_state)
        if account_state.dim() == 1:
            account_state = account_state.unsqueeze(0)
            
        # Move tensors to device
        market_state = market_state.to(self.device)
        account_state = account_state.to(self.device)
        
        return market_state, account_state

    def save_model(self, path):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
        torch.save(save_dict, path)
    
    def load_model(self, path):
        """Load model weights"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            if 'step_count' in checkpoint:
                self.step_count = checkpoint['step_count']
                
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}, starting from scratch.") 