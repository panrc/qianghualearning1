import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import os

from . import config


class TradingEnv:
    def __init__(self, file_path=config.DATA_FILE_PATH, use_cache=True):
        self.data_df = None
        self.feature_scaler = None
        self.account_info_scaler = None
        self.scaled_features = None
        self.use_cache = use_cache

        self.window_size = config.WINDOW_SIZE
        self.initial_balance = config.INITIAL_ACCOUNT_BALANCE
        self.fee_percent = config.TRANSACTION_FEE_PERCENT
        
        # Improved action space
        self.action_space = np.array(range(21))  # 0: Hold, 1-10: Buy 10%-100%, 11-20: Sell 10%-100%
        self._setup_action_mapping()

        self.current_step = 0
        self.balance = 0
        self.btc_held = 0
        self.net_worth = 0
        self.total_trades = 0
        self.buy_and_hold_btc = 0
        self.total_fees_paid = 0
        
        # Reward normalization tracking
        self.prev_net_worth = 0
        self.prev_buy_and_hold_net_worth = 0
        self.reward_scale = 100.0  # Scale factor for normalized rewards
        
        # Action tracking for analysis
        self.action_counts = np.zeros(len(self.action_space))

        self.load_data(file_path)

    def _setup_action_mapping(self):
        """Setup improved action mapping with better granularity"""
        self.action_descriptions = {}
        self.action_descriptions[0] = "HOLD"
        
        # Buy actions with improved percentages
        buy_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i, pct in enumerate(buy_percentages, 1):
            self.action_descriptions[i] = f"BUY_{int(pct*100)}%"
        
        # Sell actions with improved percentages
        sell_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i, pct in enumerate(sell_percentages, 11):
            self.action_descriptions[i] = f"SELL_{int(pct*100)}%"

    def get_action_description(self, action):
        """Get human-readable description of action"""
        return self.action_descriptions.get(action, f"UNKNOWN_{action}")

    def get_action_distribution(self):
        """Get action distribution for analysis"""
        total_actions = self.action_counts.sum()
        if total_actions == 0:
            return {desc: 0 for desc in self.action_descriptions.values()}
        
        distribution = {}
        for action, count in enumerate(self.action_counts):
            desc = self.get_action_description(action)
            distribution[desc] = (count / total_actions) * 100
        
        return distribution

    def _get_position_sizing(self, action, current_price):
        """Improved position sizing with risk management"""
        if action == 0:  # Hold
            return 0, 'HOLD'
        
        # Calculate position size with some risk management
        max_position_value = self.net_worth * 0.95  # Never use more than 95% of net worth
        
        if 1 <= action <= 10:  # Buy actions
            percentage = action * 0.1
            if self.balance <= 0:
                return 0, 'BUY_NO_CASH'
            
            # Limit buy amount to available cash
            max_spend = min(self.balance, max_position_value)
            spend_amount = max_spend * percentage
            
            return spend_amount, 'BUY'
            
        elif 11 <= action <= 20:  # Sell actions
            percentage = (action - 10) * 0.1
            if self.btc_held <= 0:
                return 0, 'SELL_NO_BTC'
            
            # Calculate BTC to sell
            btc_to_sell = self.btc_held * percentage
            return btc_to_sell, 'SELL'
        
        return 0, 'INVALID'

    def _generate_cache_path(self, file_path):
        """Generate cache file path based on input data file"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        cache_dir = os.path.dirname(config.FEATURE_CACHE_PATH)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{base_name}_features.parquet")

    def _compute_technical_indicators(self, df):
        """Compute all technical indicators for the given dataframe"""
        print("Computing technical indicators...")
        
        # Create a copy to avoid modifying the original
        base_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # Calculate Technical Indicators
        base_df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        base_df.ta.rsi(close='Close', length=14, append=True)
        base_df.ta.bbands(close='Close', length=20, std=2, append=True)
        base_df.ta.sma(close='Close', length=50, append=True)
        base_df.ta.ema(close='Close', length=21, append=True)
        base_df.ta.obv(close='Close', volume='Volume', append=True)

        # Select all feature columns and drop NaN values
        final_df = base_df[config.FEATURE_COLUMNS].copy()
        final_df.dropna(inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        print(f"Technical indicators computed. Final shape: {final_df.shape}")
        return final_df

    def _load_from_cache(self, cache_path, original_file_path):
        """Load preprocessed features from cache if available and valid"""
        if not self.use_cache or not os.path.exists(cache_path):
            return None
            
        try:
            # Check if cache is newer than original file
            cache_mtime = os.path.getmtime(cache_path)
            original_mtime = os.path.getmtime(original_file_path)
            
            if cache_mtime < original_mtime:
                print("Cache is older than original file, recomputing...")
                return None
            
            print(f"Loading preprocessed features from cache: {cache_path}")
            cached_df = pd.read_parquet(cache_path)
            print(f"Loaded cached features, shape: {cached_df.shape}")
            return cached_df
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None

    def _save_to_cache(self, df, cache_path):
        """Save preprocessed features to cache"""
        if not self.use_cache:
            return
            
        try:
            print(f"Saving preprocessed features to cache: {cache_path}")
            df.to_parquet(cache_path, compression='snappy')
            print("Features saved to cache successfully")
        except Exception as e:
            print(f"Error saving to cache: {e}")

    def load_data(self, file_path):
        """Loads a new dataset and re-initializes scalers with caching support."""
        try:
            cache_path = self._generate_cache_path(file_path)
            
            # Try to load from cache first
            cached_df = self._load_from_cache(cache_path, file_path)
            
            if cached_df is not None:
                self.data_df = cached_df
            else:
                # Load original data and compute features
                print(f"Loading original data from {file_path}")
                df = pd.read_csv(file_path, index_col='Open time', parse_dates=True)
                
                # Compute technical indicators
                self.data_df = self._compute_technical_indicators(df)
                
                # Save to cache for future use
                self._save_to_cache(self.data_df, cache_path)

            print(f"Successfully loaded data, final shape: {self.data_df.shape}")

            # Re-prepare scalers and scaled features
            self.feature_scaler, self.account_info_scaler = self._prepare_scalers()
            self.scaled_features = self._scale_features()
            return True
            
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return False

    def _prepare_scalers(self):
        feature_scaler = MinMaxScaler()
        # Fit the scaler on the feature data
        feature_scaler.fit(self.data_df[config.FEATURE_COLUMNS])

        # Create a scaler for account info. We estimate the max values.
        # Max balance is the initial balance. Max BTC can be estimated.
        # Max net worth can be initial balance * some factor
        max_btc_estimate = self.initial_balance / self.data_df['Close'].min()
        max_net_worth_estimate = self.initial_balance * 5 # Assume a 5x growth potential
        
        account_info_scaler = MinMaxScaler()
        account_info_scaler.fit(np.array([
            [0, 0, 0], # Min values for balance, btc_held, net_worth
            [self.initial_balance, max_btc_estimate, max_net_worth_estimate]
        ]))

        return feature_scaler, account_info_scaler

    def _scale_features(self):
        return self.feature_scaler.transform(self.data_df[config.FEATURE_COLUMNS])

    def _get_state(self):
        # Get the window of historical market data
        market_state = self.scaled_features[self.current_step - self.window_size + 1 : self.current_step + 1]
        
        # Get current account info and scale it
        account_info = np.array([[self.balance, self.btc_held, self.net_worth]])
        scaled_account_info = self.account_info_scaler.transform(account_info)
        
        return market_state, scaled_account_info.flatten()

    def reset(self):
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth = self.initial_balance
        self.total_trades = 0
        self.total_fees_paid = 0
        self.buy_and_hold_btc = 0
        
        # Reset action tracking
        self.action_counts = np.zeros(len(self.action_space))
        
        # Start at a random step, ensuring there's enough data for the first window
        self.current_step = np.random.randint(self.window_size, len(self.data_df) - 1)
        
        start_price = self.data_df['Close'].iloc[self.current_step]
        self.buy_and_hold_btc = self.initial_balance / start_price
        
        # Initialize previous values for reward calculation
        self.prev_net_worth = self.net_worth
        self.prev_buy_and_hold_net_worth = self.initial_balance

        return self._get_state()

    def step(self, action):
        self.current_step += 1
        
        # Track action usage
        if 0 <= action < len(self.action_counts):
            self.action_counts[action] += 1
        
        # Get the current price for calculations
        current_price = self.data_df['Close'].iloc[self.current_step]
        
        prev_net_worth = self.net_worth
        step_fee = 0
        
        # Use improved position sizing
        amount, action_type = self._get_position_sizing(action, current_price)
        
        trade_info = {'type': action_type, 'price': current_price, 'amount': 0, 'percentage': 0}

        # Execute action with improved logic
        if action_type == 'BUY' and amount > 0:
            # Fee is paid in cash
            step_fee = amount * self.fee_percent
            self.total_fees_paid += step_fee

            cash_after_fee = amount - step_fee
            btc_to_buy = cash_after_fee / current_price
            
            self.btc_held += btc_to_buy
            self.balance -= amount
            self.total_trades += 1
            
            percentage = (amount / (self.balance + amount)) * 100
            trade_info.update({
                'amount': btc_to_buy,
                'percentage': percentage,
                'fee': step_fee
            })
            
        elif action_type == 'SELL' and amount > 0:
            cash_received = amount * current_price
            
            # Fee is paid in cash
            step_fee = cash_received * self.fee_percent
            self.total_fees_paid += step_fee
            
            self.balance += cash_received - step_fee
            self.btc_held -= amount
            self.total_trades += 1
            
            percentage = (amount / (self.btc_held + amount)) * 100
            trade_info.update({
                'amount': amount,
                'percentage': percentage,
                'fee': step_fee
            })

        # Update net worth
        self.net_worth = self.balance + (self.btc_held * current_price)
        
        # Calculate buy and hold baseline net worth for this step
        buy_and_hold_net_worth = self.buy_and_hold_btc * current_price
        
        # ===== NORMALIZED REWARD CALCULATION =====
        
        # Calculate log returns for both agent and buy-and-hold
        # Use a small epsilon to avoid log(0)
        epsilon = 1e-8
        
        agent_return = np.log(max(self.net_worth, epsilon) / max(self.prev_net_worth, epsilon))
        baseline_return = np.log(max(buy_and_hold_net_worth, epsilon) / max(self.prev_buy_and_hold_net_worth, epsilon))
        
        # Base reward: scaled log return of portfolio
        base_reward = agent_return * self.reward_scale
        
        # Benchmark reward: relative performance vs buy-and-hold
        benchmark_reward = (agent_return - baseline_return) * config.BASELINE_REWARD_WEIGHT * self.reward_scale
        
        # Normalized fee penalty: fee as percentage of net worth
        fee_penalty = (step_fee / max(self.net_worth, epsilon)) * config.FEE_PENALTY_WEIGHT * self.reward_scale
        
        reward = base_reward + benchmark_reward - fee_penalty
        
        # Update previous values for next step
        self.prev_net_worth = prev_net_worth
        self.prev_buy_and_hold_net_worth = self.buy_and_hold_btc * self.data_df['Close'].iloc[self.current_step - 1]
        
        # Check if done
        done = self.current_step >= len(self.data_df) - 1

        # Bankruptcy check with normalized penalty
        if self.net_worth < 10:
            bankruptcy_penalty = config.BANKRUPTCY_PENALTY / self.initial_balance * self.reward_scale
            reward -= bankruptcy_penalty
            done = True
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'trade': trade_info,
            'current_price': current_price,
            'net_worth': self.net_worth,
            'total_trades': self.total_trades,
            'buy_and_hold_net_worth': buy_and_hold_net_worth,
            'total_fees_paid': self.total_fees_paid,
            'agent_return': agent_return,
            'baseline_return': baseline_return,
            'raw_reward': reward,
            'action_description': self.get_action_description(action),
            'action_distribution': self.get_action_distribution()
        }
        
        return next_state, reward, done, info

    def precompute_all_features(self, force_recompute=False):
        """Precompute features for all available data files (utility function)"""
        data_dir = os.path.dirname(config.DATA_FILE_PATH)
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_dir, filename)
                cache_path = self._generate_cache_path(file_path)
                
                if force_recompute or not os.path.exists(cache_path):
                    print(f"Precomputing features for {filename}")
                    try:
                        df = pd.read_csv(file_path, index_col='Open time', parse_dates=True)
                        processed_df = self._compute_technical_indicators(df)
                        self._save_to_cache(processed_df, cache_path)
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                else:
                    print(f"Features already cached for {filename}")
        
        print("Feature precomputation completed") 