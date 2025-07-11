import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from . import config

class NoisyLinear(nn.Module):
    """Noisy network layer for automatic exploration"""
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise parameters (not learnable)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

class DoubleDuelingDQN_LSTM(nn.Module):
    def __init__(self, use_noisy=False):
        super(DoubleDuelingDQN_LSTM, self).__init__()
        self.use_noisy = use_noisy
        
        # LSTM layer to process sequential market data
        self.lstm = nn.LSTM(
            input_size=config.INPUT_FEATURES,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Feature extraction layers
        concatenated_size = config.HIDDEN_SIZE + 3
        self.feature_layer = nn.Linear(concatenated_size, 128)
        self.feature_layer2 = nn.Linear(128, 64)
        
        # Dueling DQN: separate value and advantage streams
        if use_noisy:
            self.value_stream = nn.Sequential(
                NoisyLinear(64, 32),
                nn.ReLU(),
                NoisyLinear(32, 1)
            )
            self.advantage_stream = nn.Sequential(
                NoisyLinear(64, 32),
                nn.ReLU(),
                NoisyLinear(32, config.NUM_ACTIONS)
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, config.NUM_ACTIONS)
            )

    def forward(self, market_state, account_state, hidden_state=None):
        # market_state shape: [batch_size, window_size, num_features]
        # account_state shape: [batch_size, 3]
        
        batch_size = market_state.size(0)
        
        # Pass market data through LSTM with optional hidden state
        if hidden_state is not None:
            lstm_out, (h_n, c_n) = self.lstm(market_state, hidden_state)
        else:
            lstm_out, (h_n, c_n) = self.lstm(market_state)
        
        # Get the last hidden state from the last layer
        lstm_output = h_n[-1]
        
        # Concatenate LSTM output with account information
        combined_features = torch.cat((lstm_output, account_state), dim=1)
        
        # Pass through feature extraction layers
        x = F.relu(self.feature_layer(combined_features))
        x = F.relu(self.feature_layer2(x))
        
        # Dueling DQN: compute value and advantage separately
        value = self.value_stream(x)  # [batch_size, 1]
        advantage = self.advantage_stream(x)  # [batch_size, num_actions]
        
        # Combine value and advantage using dueling formula
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values, (h_n, c_n)
    
    def reset_noise(self):
        """Reset noise for noisy networks"""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

# Legacy alias for backward compatibility
DQN_LSTM = DoubleDuelingDQN_LSTM 