import time
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
import os

from . import config

class Trainer(QThread):
    new_step_data = pyqtSignal(dict)
    new_episode_stats = pyqtSignal(dict)
    training_status_update = pyqtSignal(str)

    def __init__(self, agent, env, is_validation_mode=False):
        super().__init__()
        self.agent = agent
        self.env = env
        self.is_validation_mode = is_validation_mode
        self.should_stop = False
        self.is_paused = False
        self.best_reward = float('-inf')
        self.best_net_worth = 0
        
        # Training statistics
        self.total_steps = 0
        self.episode_count = 0
        
        # AMP training
        self.use_amp = config.USE_AMP and config.DEVICE == "cuda"
        if self.use_amp:
            # 修复 GradScaler 警告
            self.scaler = torch.amp.GradScaler(device='cuda')
            print("Using Automatic Mixed Precision training")
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS

    def run(self):
        self.training_status_update.emit("Training Started" if not self.is_validation_mode else "Validation Started")
        
        while not self.should_stop:
            if self.is_paused:
                time.sleep(0.1)
                continue
                
            self.episode_count += 1
            episode_reward, avg_loss = self._run_episode()
            
            if not self.should_stop:
                self._handle_episode_end((episode_reward, avg_loss))
        
        self.training_status_update.emit("Training Stopped" if not self.is_validation_mode else "Validation Stopped")

    def _run_episode(self):
        """Run a single episode and return total reward"""
        state = self.env.reset()
        
        # Reset LSTM hidden states at episode start
        if not self.is_validation_mode:
            self.agent.reset_hidden_states()
        
        total_reward = 0
        done = False
        step_count = 0
        episode_losses = []
        
        while not done and not self.should_stop:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            # Select action (epsilon is handled internally by agent)
            action = self.agent.select_action(state, epsilon=0.0 if self.is_validation_mode else None)
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            self.total_steps += 1
            
            # Store transition and optimize (only during training)
            loss = None
            if not self.is_validation_mode:
                # Store transition in replay buffer
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Optimize model with gradient accumulation
                if self.total_steps % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        loss = self._optimize_with_amp()
                    else:
                        loss = self.agent.optimize_model()
                    
                    if loss is not None:
                        episode_losses.append(loss)
                
                # Update target network periodically
                if self.total_steps % config.TARGET_UPDATE_FREQUENCY == 0:
                    self.agent.update_target_net()
                    self.training_status_update.emit(f"Target network updated at step {self.total_steps}")
                
                # Update epsilon
                self.agent.update_epsilon()
            
            # Emit step data for GUI updates
            step_data = {
                'step': self.env.current_step,
                'price': info['current_price'],
                'net_worth': info['net_worth'],
                'buy_and_hold_net_worth': info['buy_and_hold_net_worth'],
                'action': action,
                'trade': info['trade'],
                'cash': self.env.balance,
                'btc_held': self.env.btc_held,
                'total_trades': self.env.total_trades,
                'total_fees_paid': info['total_fees_paid'],
                'epsilon': self.agent.epsilon,
                'loss': loss,
                'agent_return': info.get('agent_return', 0),
                'baseline_return': info.get('baseline_return', 0),
                'raw_reward': info.get('raw_reward', reward)
            }
            self.new_step_data.emit(step_data)
            
            state = next_state
            step_count += 1
            
            # Small delay to allow GUI updates
            time.sleep(0.001)
        
        return total_reward, np.mean(episode_losses) if episode_losses else 0.0

    def _optimize_with_amp(self):
        """Optimize model with Automatic Mixed Precision"""
        if len(self.agent.memory) < config.BATCH_SIZE:
            return None
        
        # Get batch data
        if self.agent.use_prioritized_replay:
            batch_data = self.agent.memory.sample(config.BATCH_SIZE)
            (market_states_t, account_states_t, actions_t, rewards_t, 
             next_market_states_t, next_account_states_t, dones_t, 
             is_weights, batch_idx) = batch_data
        else:
            market_states_t, account_states_t, actions_t, rewards_t, next_market_states_t, next_account_states_t, dones_t = \
                self.agent.memory.sample(config.BATCH_SIZE)
            is_weights, batch_idx = None, None
        
        # Reset noise for noisy networks
        if self.agent.use_noisy_net:
            self.agent.policy_net.reset_noise()
            self.agent.target_net.reset_noise()
        
        # 修复 autocast 警告和数据类型不匹配
        with torch.amp.autocast(device_type='cuda'):
            # Current Q values
            q_values, _ = self.agent.policy_net(market_states_t, account_states_t)
            state_action_values = q_values.gather(1, actions_t.unsqueeze(1))
            
            # Next state Q values using Double DQN
            next_state_values = torch.zeros(config.BATCH_SIZE, device=self.agent.device)
            non_final_mask = ~dones_t
            
            if non_final_mask.sum() > 0:
                with torch.no_grad():
                    non_final_next_market = next_market_states_t[non_final_mask]
                    non_final_next_account = next_account_states_t[non_final_mask]
                    
                    # Double DQN
                    next_q_policy, _ = self.agent.policy_net(non_final_next_market, non_final_next_account)
                    next_actions = next_q_policy.max(1)[1]
                    
                    next_q_target, _ = self.agent.target_net(non_final_next_market, non_final_next_account)
                    next_state_values[non_final_mask] = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute expected Q values
            expected_state_action_values = rewards_t + (config.GAMMA * next_state_values)
            
            # Compute loss
            if is_weights is not None:
                # Prioritized replay loss
                td_errors = state_action_values.squeeze(1) - expected_state_action_values
                loss = (is_weights * F.smooth_l1_loss(
                    state_action_values.squeeze(1), 
                    expected_state_action_values, 
                    reduction='none'
                )).mean()
            else:
                loss = F.smooth_l1_loss(
                    state_action_values.squeeze(1), 
                    expected_state_action_values
                )
        
        # Backward pass with gradient scaling
        self.agent.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.agent.optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), config.GRAD_CLIP_NORM)
        
        self.scaler.step(self.agent.optimizer)
        self.scaler.update()
        
        # Update priorities for prioritized replay
        if is_weights is not None and batch_idx is not None:
            with torch.no_grad():
                td_errors_abs = td_errors.abs().detach().cpu().numpy()
                self.agent.memory.update_priorities(batch_idx, td_errors_abs)
        
        return loss.item()

    def _handle_episode_end(self, episode_data):
        """Handle end of episode statistics and model saving"""
        total_reward, avg_loss = episode_data
        
        # Calculate profit/loss
        profit_loss = self.env.net_worth - config.INITIAL_ACCOUNT_BALANCE
        profit_percentage = (profit_loss / config.INITIAL_ACCOUNT_BALANCE) * 100
        
        # Emit episode stats
        episode_stats = {
            'episode': self.episode_count,
            'total_reward': total_reward,
            'avg_loss': avg_loss,
            'final_balance': self.env.net_worth,
            'profit_loss': profit_loss,
            'profit_percentage': profit_percentage,
            'total_trades': self.env.total_trades,
            'total_fees_paid': self.env.total_fees_paid,
            'epsilon': self.agent.epsilon,
            'total_steps': self.total_steps
        }
        self.new_episode_stats.emit(episode_stats)
        
        # Save best model (only during training)
        if not self.is_validation_mode:
            model_saved = False
            
            # Save if better reward
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                model_saved = True
            
            # Also save if better net worth
            if self.env.net_worth > self.best_net_worth:
                self.best_net_worth = self.env.net_worth
                model_saved = True
            
            if model_saved:
                self.agent.save_model(config.BEST_MODEL_PATH)
                self.training_status_update.emit(
                    f"New best model saved! Reward: {total_reward:.2f}, "
                    f"Net Worth: ${self.env.net_worth:.2f}, Profit: {profit_percentage:.1f}%"
                )
            
            # Regular checkpoint saving
            if self.episode_count % config.CHECKPOINT_INTERVAL == 0:
                checkpoint_path = os.path.join(config.SAVED_MODELS_DIR, f"checkpoint_ep_{self.episode_count}.pth")
                self.agent.save_model(checkpoint_path)
                self.training_status_update.emit(f"Checkpoint saved: {checkpoint_path}")
        
        # Log episode completion
        mode = "Validation" if self.is_validation_mode else "Training"
        self.training_status_update.emit(
            f"{mode} Episode {self.episode_count} completed. "
            f"Reward: {total_reward:.2f}, Net Worth: ${self.env.net_worth:.2f}, "
            f"Profit: {profit_percentage:.1f}%, ε: {self.agent.epsilon:.3f}"
        )
        
        # Log training statistics periodically
        if not self.is_validation_mode and self.episode_count % config.LOG_INTERVAL == 0:
            self._log_training_stats(total_reward, avg_loss)

    def _log_training_stats(self, total_reward, avg_loss):
        """Log detailed training statistics"""
        stats_msg = (
            f"=== Training Stats (Episode {self.episode_count}) ===\n"
            f"Total Steps: {self.total_steps}\n"
            f"Epsilon: {self.agent.epsilon:.4f}\n"
            f"Average Loss: {avg_loss:.6f}\n"
            f"Best Reward: {self.best_reward:.2f}\n"
            f"Best Net Worth: ${self.best_net_worth:.2f}\n"
            f"Memory Size: {len(self.agent.memory)}\n"
        )
        self.training_status_update.emit(stats_msg)

    def pause_training(self):
        self.is_paused = True

    def resume_training(self):
        self.is_paused = False

    def stop_training(self):
        self.should_stop = True 