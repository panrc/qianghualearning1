import sys
from collections import deque
import os

import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QGridLayout, QFileDialog, QCheckBox)
from PyQt6.QtCore import Qt

from . import config
from .agent import Agent
from .trading_env import TradingEnv
from .trainer import Trainer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced RL Crypto Trading Bot - Double Dueling DQN with PER")
        self.setGeometry(100, 100, 1800, 1000)

        # Initialize Agent and Env with improved features
        self.agent = Agent(
            use_prioritized_replay=config.USE_PRIORITIZED_REPLAY,
            use_noisy_net=config.USE_NOISY_NETWORKS
        )
        self.env = TradingEnv()
        self.trainer_thread = None

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self._setup_ui()
        self._connect_actions()

    def _setup_ui(self):
        # --- Left Panel (Main Charts) ---
        left_panel = QVBoxLayout()

        # Price and Asset Value charts
        self.price_plot_widget = pg.PlotWidget(title="BTC/USDT Price with Trade Signals")
        self.price_plot_widget.showGrid(x=True, y=True)
        self.price_curve = self.price_plot_widget.plot(pen='y', name="BTC Price")
        self.buy_scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(0, 255, 0, 200), symbol='t', pen=None)
        self.sell_scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 0, 0, 200), symbol='t1', pen=None)
        self.price_plot_widget.addItem(self.buy_scatter)
        self.price_plot_widget.addItem(self.sell_scatter)
        
        self.asset_plot_widget = pg.PlotWidget(title="Portfolio Performance Comparison")
        self.asset_plot_widget.addLegend()
        self.asset_plot_widget.showGrid(x=True, y=True)
        self.asset_curve = self.asset_plot_widget.plot(pen='c', name="Agent Portfolio")
        self.baseline_curve = self.asset_plot_widget.plot(pen=pg.mkPen('w', style=Qt.PenStyle.DashLine), name="Buy & Hold Baseline")
        
        # Reward curve chart
        self.reward_plot_widget = pg.PlotWidget(title="Episode Rewards & Returns")
        self.reward_plot_widget.addLegend()
        self.reward_plot_widget.showGrid(x=True, y=True)
        self.reward_curve = self.reward_plot_widget.plot(pen='g', name="Episode Reward")
        self.return_curve = self.reward_plot_widget.plot(pen='m', name="Agent Return")
        
        left_panel.addWidget(self.price_plot_widget)
        left_panel.addWidget(self.asset_plot_widget)
        left_panel.addWidget(self.reward_plot_widget)

        # --- Right Panel (Controls, Stats, Small Charts) ---
        right_panel = QVBoxLayout()
        
        # Controls with advanced options
        control_layout = QVBoxLayout()
        
        # Main controls
        main_control_layout = QHBoxLayout()
        self.load_dataset_btn = QPushButton("Load Dataset")
        self.load_weights_btn = QPushButton("Load Weights")
        self.start_btn = QPushButton("Start Training")
        self.validate_btn = QPushButton("Start Validation")
        self.pause_btn = QPushButton("Pause Training")
        self.save_weights_btn = QPushButton("Save Weights")
        self.pause_btn.setDisabled(True)
        self.save_weights_btn.setDisabled(True)
        
        main_control_layout.addWidget(self.load_dataset_btn)
        main_control_layout.addWidget(self.load_weights_btn)
        main_control_layout.addWidget(self.start_btn)
        main_control_layout.addWidget(self.validate_btn)
        main_control_layout.addWidget(self.pause_btn)
        main_control_layout.addWidget(self.save_weights_btn)
        
        # Advanced options
        advanced_layout = QHBoxLayout()
        self.prioritized_replay_checkbox = QCheckBox("Prioritized Replay")
        self.prioritized_replay_checkbox.setChecked(config.USE_PRIORITIZED_REPLAY)
        self.noisy_net_checkbox = QCheckBox("Noisy Networks")
        self.noisy_net_checkbox.setChecked(config.USE_NOISY_NETWORKS)
        self.amp_checkbox = QCheckBox("Mixed Precision")
        self.amp_checkbox.setChecked(config.USE_AMP)
        
        advanced_layout.addWidget(self.prioritized_replay_checkbox)
        advanced_layout.addWidget(self.noisy_net_checkbox)
        advanced_layout.addWidget(self.amp_checkbox)
        
        control_layout.addLayout(main_control_layout)
        control_layout.addLayout(advanced_layout)
        right_panel.addLayout(control_layout)

        # Enhanced Stats Labels
        stats_layout = QGridLayout()
        self.status_label = QLabel("Status: Idle")
        self.dataset_label = QLabel(f"Dataset: {os.path.basename(config.DATA_FILE_PATH)}")
        self.episode_label = QLabel("Episode: 0")
        self.step_label = QLabel("Step: 0")
        self.profit_label = QLabel("P/L: $0.00")
        self.profit_pct_label = QLabel("P/L %: 0.00%")
        self.net_worth_label = QLabel(f"Net Worth: ${config.INITIAL_ACCOUNT_BALANCE:.2f}")
        self.trades_label = QLabel("Trades: 0")
        self.fees_label = QLabel("Total Fees: $0.00")
        self.epsilon_label = QLabel("Epsilon: 1.0")
        self.cash_label = QLabel(f"Cash: ${config.INITIAL_ACCOUNT_BALANCE:.2f}")
        self.btc_held_label = QLabel("BTC Held: 0.00000")
        self.max_drawdown_label = QLabel("Max Drawdown: 0.00%")
        self.loss_label = QLabel("Current Loss: N/A")
        self.memory_size_label = QLabel("Memory Size: 0")
        self.action_label = QLabel("Last Action: HOLD")
        
        stats_layout.addWidget(self.status_label, 0, 0, 1, 2)
        stats_layout.addWidget(self.dataset_label, 1, 0, 1, 2)
        stats_layout.addWidget(self.episode_label, 2, 0)
        stats_layout.addWidget(self.step_label, 2, 1)
        stats_layout.addWidget(self.profit_label, 3, 0)
        stats_layout.addWidget(self.profit_pct_label, 3, 1)
        stats_layout.addWidget(self.net_worth_label, 4, 0)
        stats_layout.addWidget(self.trades_label, 4, 1)
        stats_layout.addWidget(self.fees_label, 5, 0)
        stats_layout.addWidget(self.epsilon_label, 5, 1)
        stats_layout.addWidget(self.cash_label, 6, 0)
        stats_layout.addWidget(self.btc_held_label, 6, 1)
        stats_layout.addWidget(self.max_drawdown_label, 7, 0)
        stats_layout.addWidget(self.loss_label, 7, 1)
        stats_layout.addWidget(self.memory_size_label, 8, 0)
        stats_layout.addWidget(self.action_label, 8, 1)
        right_panel.addLayout(stats_layout)
        
        # Enhanced Small Charts
        small_charts_layout = QGridLayout()
        
        self.balance_plot = pg.PlotWidget(title="Balance per Episode ($)")
        self.balance_curve = self.balance_plot.plot(pen='g')
        
        self.epsilon_plot = pg.PlotWidget(title="Exploration Rate (Epsilon)")
        self.epsilon_curve = self.epsilon_plot.plot(pen='m')
        
        # Improved action distribution plot
        self.action_dist_plot = pg.PlotWidget(title="Action Distribution")
        self.action_bars = pg.BarGraphItem(x=[0, 1, 2], height=[0, 0, 0], width=0.5, brushes=['gray', 'green', 'red'])
        self.action_dist_plot.addItem(self.action_bars)
        self.action_dist_plot.getAxis('bottom').setTicks([[(0, 'Hold'), (1, 'Buy'), (2, 'Sell')]])

        self.loss_plot = pg.PlotWidget(title="Training Loss")
        self.loss_curve = self.loss_plot.plot(pen='r')

        # New charts for enhanced features
        self.returns_plot = pg.PlotWidget(title="Agent vs Baseline Returns")
        self.returns_plot.addLegend()
        self.agent_returns_curve = self.returns_plot.plot(pen='c', name="Agent Returns")
        self.baseline_returns_curve = self.returns_plot.plot(pen='w', name="Baseline Returns")

        self.td_error_plot = pg.PlotWidget(title="TD Error (PER)")
        self.td_error_curve = self.td_error_plot.plot(pen='orange')

        small_charts_layout.addWidget(self.balance_plot, 0, 0)
        small_charts_layout.addWidget(self.epsilon_plot, 0, 1)
        small_charts_layout.addWidget(self.action_dist_plot, 1, 0)
        small_charts_layout.addWidget(self.loss_plot, 1, 1)
        small_charts_layout.addWidget(self.returns_plot, 2, 0)
        small_charts_layout.addWidget(self.td_error_plot, 2, 1)
        right_panel.addLayout(small_charts_layout)

        # Add panels to main layout
        self.main_layout.addLayout(left_panel, stretch=3)
        self.main_layout.addLayout(right_panel, stretch=2)

        # Enhanced data deques for plotting
        self.time_data = deque(maxlen=config.MAX_CHART_POINTS)
        self.price_data = deque(maxlen=config.MAX_CHART_POINTS)
        self.asset_data = deque(maxlen=config.MAX_CHART_POINTS)
        self.baseline_data = deque(maxlen=config.MAX_CHART_POINTS)
        self.epsilon_plot_data = deque(maxlen=config.MAX_CHART_POINTS)
        self.loss_plot_data = deque(maxlen=config.MAX_CHART_POINTS)
        self.agent_returns_data = deque(maxlen=config.MAX_CHART_POINTS)
        self.baseline_returns_data = deque(maxlen=config.MAX_CHART_POINTS)
        self.td_error_data = deque(maxlen=config.MAX_CHART_POINTS)

        self.episode_balance_data = []
        self.episode_reward_data = []
        self.episode_return_data = []
        self.action_counts = [0, 0, 0]  # Hold, Buy (grouped), Sell (grouped)
        
        # Tracking variables for enhanced metrics
        self.peak_net_worth = config.INITIAL_ACCOUNT_BALANCE
        self.max_drawdown = 0.0

    def _connect_actions(self):
        self.start_btn.clicked.connect(self.start_training)
        self.validate_btn.clicked.connect(self.start_validation)
        self.pause_btn.clicked.connect(self.pause_training)
        self.load_weights_btn.clicked.connect(self.load_weights)
        self.save_weights_btn.clicked.connect(self.save_weights)
        self.load_dataset_btn.clicked.connect(self.load_dataset)

    def start_training(self):
        self.start_thread(is_validation_mode=False)

    def start_validation(self):
        self.start_thread(is_validation_mode=True)
    
    def start_thread(self, is_validation_mode):
        """Start training or validation thread"""
        # Update agent settings based on checkboxes
        self.agent = Agent(
            use_prioritized_replay=self.prioritized_replay_checkbox.isChecked(),
            use_noisy_net=self.noisy_net_checkbox.isChecked()
        )
        
        # Initialize trainer
        self.trainer_thread = Trainer(
            agent=self.agent,
            env=self.env,
            is_validation_mode=is_validation_mode
        )
        
        # Connect signals
        self.trainer_thread.new_step_data.connect(self.update_step_plots)
        self.trainer_thread.new_episode_stats.connect(self.update_episode_plots)
        self.trainer_thread.training_status_update.connect(self.update_status)
        
        # Update UI state
        self.start_btn.setDisabled(True)
        self.validate_btn.setDisabled(True)
        self.pause_btn.setEnabled(True)
        self.save_weights_btn.setEnabled(True)
        self.load_weights_btn.setDisabled(True)
        self.load_dataset_btn.setDisabled(True)
        
        # Reset tracking variables
        self.peak_net_worth = config.INITIAL_ACCOUNT_BALANCE
        self.max_drawdown = 0.0
        self.action_counts = [0, 0, 0]
        
        # Start thread
        self.trainer_thread.start()

    def pause_training(self):
        if self.trainer_thread and self.trainer_thread.isRunning():
            if self.pause_btn.text() == "Pause Training":
                self.trainer_thread.pause_training()
                self.pause_btn.setText("Resume Training")
            else:
                self.trainer_thread.resume_training()
                self.pause_btn.setText("Pause Training")

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "CSV Files (*.csv)")
        if path:
            if self.env.load_data(path):
                self.dataset_label.setText(f"Dataset: {os.path.basename(path)}")
                self.update_status(f"Loaded dataset: {os.path.basename(path)}")
            else:
                self.update_status(f"Failed to load dataset: {os.path.basename(path)}")

    def load_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model Weights", config.SAVED_MODELS_DIR, "PyTorch Models (*.pth)")
        if path:
            self.agent.load_model(path)
            self.update_status(f"Loaded weights from {os.path.basename(path)}")
    
    def save_weights(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Model Weights", config.SAVED_MODELS_DIR, "PyTorch Models (*.pth)")
        if path:
            self.agent.save_model(path)
            self.update_status(f"Saved weights to {os.path.basename(path)}")

    def update_status(self, status):
        self.status_label.setText(f"Status: {status}")

    def update_step_plots(self, data):
        current_step = data['step']
        self.time_data.append(current_step)
        self.price_data.append(data['price'])
        self.asset_data.append(data['net_worth'])
        self.baseline_data.append(data['buy_and_hold_net_worth'])
        
        # Update main plots
        self.price_curve.setData(list(self.time_data), list(self.price_data))
        self.asset_curve.setData(list(self.time_data), list(self.asset_data))
        self.baseline_curve.setData(list(self.time_data), list(self.baseline_data))

        # Add returns data if available
        if 'agent_return' in data and 'baseline_return' in data:
            self.agent_returns_data.append(data['agent_return'])
            self.baseline_returns_data.append(data['baseline_return'])
            self.agent_returns_curve.setData(list(self.time_data), list(self.agent_returns_data))
            self.baseline_returns_curve.setData(list(self.time_data), list(self.baseline_returns_data))

        # Dynamically set the X-axis range to show the most recent points
        if len(self.time_data) > 1:
            x_min = max(self.time_data[0], current_step - config.MAX_CHART_POINTS)
            x_max = current_step
            self.price_plot_widget.setXRange(x_min, x_max, padding=0.05)
            self.asset_plot_widget.setXRange(x_min, x_max, padding=0.05)
            self.reward_plot_widget.setXRange(max(0, len(self.episode_reward_data) - 100), 
                                               len(self.episode_reward_data), padding=0.05)
        
        # Update action distribution
        if data['action'] == 0:
            self.action_counts[0] += 1  # Hold
        elif 1 <= data['action'] <= 10:
            self.action_counts[1] += 1  # Buy
        elif 11 <= data['action'] <= 20:
            self.action_counts[2] += 1  # Sell
            
        self.action_bars.setOpts(height=self.action_counts)
        
        # Update epsilon plot
        if data['epsilon']:
            self.epsilon_plot_data.append(data['epsilon'])
            self.epsilon_curve.setData(list(self.epsilon_plot_data))
            self.epsilon_label.setText(f"Epsilon: {data['epsilon']:.4f}")

        # Update loss plot
        if data['loss'] is not None:
            self.loss_plot_data.append(data['loss'])
            self.loss_curve.setData(list(self.loss_plot_data))
            self.loss_label.setText(f"Current Loss: {data['loss']:.6f}")

        # Add trade markers
        if data['trade']['type'] == 'BUY':
            self.buy_scatter.addPoints([{'pos': (data['step'], data['price']), 'data': 1}])
        elif data['trade']['type'] == 'SELL':
            self.sell_scatter.addPoints([{'pos': (data['step'], data['price']), 'data': 1}])

        # Update enhanced labels
        self.step_label.setText(f"Step: {current_step}")
        self.net_worth_label.setText(f"Net Worth: ${data['net_worth']:.2f}")
        self.cash_label.setText(f"Cash: ${data['cash']:.2f}")
        self.btc_held_label.setText(f"BTC Held: {data['btc_held']:.5f}")
        self.trades_label.setText(f"Trades: {data['total_trades']}")
        self.fees_label.setText(f"Total Fees: ${data['total_fees_paid']:.2f}")
        self.memory_size_label.setText(f"Memory Size: {len(self.agent.memory)}")
        
        # Update action description
        if 'action_description' in data:
            self.action_label.setText(f"Last Action: {data['action_description']}")
        
        # Calculate and update max drawdown
        current_net_worth = data['net_worth']
        self.peak_net_worth = max(self.peak_net_worth, current_net_worth)
        
        current_drawdown = (self.peak_net_worth - current_net_worth) / self.peak_net_worth * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        self.max_drawdown_label.setText(f"Max Drawdown: {self.max_drawdown:.2f}%")

    def update_episode_plots(self, data):
        """Called at the end of each episode."""
        self.episode_label.setText(f"Episode: {data['episode']}")
        self.net_worth_label.setText(f"Net Worth: ${data['final_balance']:.2f}")
        self.profit_label.setText(f"P/L: ${data['profit_loss']:.2f}")
        
        if 'profit_percentage' in data:
            self.profit_pct_label.setText(f"P/L %: {data['profit_percentage']:.2f}%")

        # Reset trackers for the new episode
        self.peak_net_worth = config.INITIAL_ACCOUNT_BALANCE
        self.max_drawdown = 0.0

        # Update episode-level plots
        self.episode_balance_data.append(data['final_balance'])
        self.balance_curve.setData(self.episode_balance_data)
        
        if 'total_reward' in data:
            self.episode_reward_data.append(data['total_reward'])
            self.reward_curve.setData(self.episode_reward_data)
        
        # Clear some data for new episode
        self.buy_scatter.clear()
        self.sell_scatter.clear()

    def closeEvent(self, event):
        if self.trainer_thread and self.trainer_thread.isRunning():
            self.trainer_thread.stop_training()
            self.trainer_thread.wait()
        event.accept()

def run_gui():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 