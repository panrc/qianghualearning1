import torch

# --- Device Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- Data Config ---
DATA_FILE_PATH = "data/BTCUSDT_5m_2years.csv"
# The columns from the CSV file to be used as features
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'RSI_14',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
    'SMA_50',
    'EMA_21',
    'OBV'
]

# --- Environment Config ---
INITIAL_ACCOUNT_BALANCE = 10000.0
TRANSACTION_FEE_PERCENT = 0.001  # 0.1% transaction fee
WINDOW_SIZE = 60  # Number of past time steps to consider for the state

# --- Model Hyperparameters ---
INPUT_FEATURES = len(FEATURE_COLUMNS)
HIDDEN_SIZE = 128  # LSTM hidden size
NUM_ACTIONS = 21  # 0: Hold, 1-10: Buy 10%-100%, 11-20: Sell 10%-100%

# --- Training Hyperparameters ---
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor for future rewards
EPSILON_START = 0.9  # Starting value of epsilon (increased for better exploration)
EPSILON_MIN = 0.01  # Minimum value of epsilon (reduced for better exploitation)
EPSILON_DECAY = 0.9995  # Epsilon decay rate per step (slower decay)
TARGET_UPDATE_FREQUENCY = 1000  # How often to update the target network (in steps)
LEARNING_RATE = 1e-4  # Reduced learning rate for stability
REPLAY_MEMORY_SIZE = 100000  # Increased replay buffer size

# --- Advanced Training Options ---
USE_PRIORITIZED_REPLAY = True  # Enable Prioritized Experience Replay
USE_NOISY_NETWORKS = False  # Enable Noisy Networks for exploration
USE_DOUBLE_DQN = True  # Enable Double DQN (always enabled in our implementation)
USE_DUELING_DQN = True  # Enable Dueling DQN (always enabled in our implementation)
USE_AMP = True  # Enable Automatic Mixed Precision training
GRADIENT_ACCUMULATION_STEPS = 1  # Number of steps to accumulate gradients

# --- Prioritized Replay Config ---
PER_ALPHA = 0.6  # How much prioritization is used
PER_BETA_START = 0.4  # Initial importance sampling weight
PER_BETA_FRAMES = 100000  # Number of frames over which beta increases to 1.0

# --- Noisy Networks Config ---
NOISY_STD_INIT = 0.4  # Initial std for noisy layers

# --- Performance Optimization ---
USE_TORCH_COMPILE = False  # Enable torch.compile (requires PyTorch 2.0+)
NUM_WORKERS = 4  # Number of workers for data loading (if applicable)
PIN_MEMORY = True  # Pin memory for faster GPU transfer

# --- GUI Config ---
CHART_UPDATE_INTERVAL_MS = 50  # Update GUI charts every 50ms
MAX_CHART_POINTS = 1000  # Increased max data points for better visualization
GUI_REFRESH_RATE = 30  # GUI refresh rate in FPS

# --- File Paths ---
SAVED_MODELS_DIR = "saved_models"
BEST_MODEL_NAME = "best_model.pth"
BEST_MODEL_PATH = f"{SAVED_MODELS_DIR}/{BEST_MODEL_NAME}"
CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N episodes

# --- Reward Tuning ---
BANKRUPTCY_PENALTY = 1000.0  # Large penalty for going bankrupt
MISSED_PROFIT_PENALTY_WEIGHT = 1.0  # Penalty coefficient for missing out on price increases
BASELINE_REWARD_WEIGHT = 10.0  # Increased weight for benchmark comparison
FEE_PENALTY_WEIGHT = 5.0  # Increased penalty weight for transaction fees

# --- Data Preprocessing ---
PRECOMPUTE_FEATURES = True  # Whether to precompute technical indicators
FEATURE_CACHE_PATH = "data/preprocessed_features.parquet"  # Cache file for preprocessed features

# --- Logging Config ---
LOG_INTERVAL = 100  # Log training statistics every N steps
SAVE_INTERVAL = 5000  # Save model every N steps
TENSORBOARD_LOG_DIR = "logs/tensorboard"  # TensorBoard log directory
ENABLE_WANDB = False  # Enable Weights & Biases logging
WANDB_PROJECT = "crypto-trading-rl"  # W&B project name

# --- Training Stability ---
GRAD_CLIP_NORM = 1.0  # Gradient clipping norm
WEIGHT_DECAY = 1e-5  # L2 regularization
DROPOUT_RATE = 0.2  # Dropout rate for regularization

# --- Multi-Environment Training (Future Enhancement) ---
NUM_ENVS = 1  # Number of parallel environments
ASYNC_TRAINING = False  # Enable asynchronous training

# --- Action Space Improvements ---
ACTION_SPACE_TYPE = "discrete"  # "discrete" or "continuous"
# For continuous action space (future enhancement)
ACTION_LOW = -1.0  # Minimum action value
ACTION_HIGH = 1.0  # Maximum action value 