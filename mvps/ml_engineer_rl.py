import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from collections import deque
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import gymnasium as gym
from gymnasium import spaces

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg

########################################
# Environment (MLEnvironment)
########################################

class MLArchitecture:
    def __init__(self):
        self.layers = []
        self.optimizer_config = {}
        self.loss_function = None
        
    def add_layer(self, layer_type: str, config: Dict[str, Any]):
        # Only support Dense
        if layer_type != 'Dense':
            raise ValueError("Only 'Dense' layers are supported in this MVP.")
        self.layers.append({"type": layer_type, "config": config})
        
    def set_optimizer(self, optimizer_type: str, config: Dict[str, Any]):
        if optimizer_type not in ['adam', 'sgd', 'rmsprop']:
            raise ValueError("Unsupported optimizer type.")
        self.optimizer_config = {"type": optimizer_type, "config": config}
        
    def set_loss(self, loss_type: str):
        if loss_type not in ['cross_entropy', 'mse']:
            raise ValueError("Unsupported loss type.")
        self.loss_function = loss_type
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layers": self.layers,
            "optimizer": self.optimizer_config,
            "loss": self.loss_function
        }

class ModelTester:
    def __init__(self, input_size=20, n_classes=2, test_samples=1000):
        self.input_size = input_size
        self.n_classes = n_classes
        self.test_samples = test_samples
        self._generate_data()
        self.history = deque(maxlen=100)
        
    def _generate_data(self):
        X, y = make_classification(
            n_samples=self.test_samples,
            n_features=self.input_size,
            n_classes=self.n_classes,
            n_clusters_per_class=2,
            random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
    def update_curriculum(self, episode: int):
        # Increase complexity after 50 episodes for demonstration
        if episode > 50:
            self.test_samples = 2000
            self.input_size = 40
            self._generate_data()
        
    def test_architecture(self, architecture: MLArchitecture) -> Dict[str, float]:
        try:
            model = self._build_model(architecture)
            
            X_train = torch.FloatTensor(self.X_train)
            y_train = torch.LongTensor(self.y_train)
            X_test = torch.FloatTensor(self.X_test)
            y_test = torch.LongTensor(self.y_test)
            
            criterion = self._get_loss_function(architecture.loss_function)
            optimizer = self._get_optimizer(model, architecture.optimizer_config)
            
            epochs = 10
            batch_size = 32
            
            initial_loss = self._calculate_loss(model, X_train, y_train, criterion)
            start_time = time.time()
            
            for _ in range(epochs):
                model.train()
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            training_time = time.time() - start_time
            final_loss = self._calculate_loss(model, X_train, y_train, criterion)
            
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                predictions = torch.argmax(test_outputs, dim=1).numpy()
                
            metrics = {
                'accuracy': float(accuracy_score(self.y_test, predictions)),
                'precision': float(precision_score(self.y_test, predictions, average='weighted', zero_division=0)),
                'recall': float(recall_score(self.y_test, predictions, average='weighted', zero_division=0)),
                'f1': float(f1_score(self.y_test, predictions, average='weighted', zero_division=0)),
                'training_time': training_time,
                'convergence_rate': float(abs(final_loss - initial_loss))
            }
            
            self.history.append({
                'architecture': architecture.to_dict(),
                'metrics': metrics
            })
            
            return metrics
        except Exception as e:
            print(f"Error testing architecture: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'training_time': 0.0,
                'convergence_rate': 1.0
            }
    
    def _build_model(self, architecture: MLArchitecture) -> nn.Module:
        layers = []
        input_size = self.input_size
        
        for layer in architecture.layers:
            if layer['type'] == 'Dense':
                units = layer['config']['units']
                activation = layer['config']['activation']
                layers.append(nn.Linear(input_size, units))
                layers.extend(self._get_activation_layer(activation))
                input_size = units
        
        layers.append(nn.Linear(input_size, self.n_classes))
        layers.append(nn.Softmax(dim=1))
        
        return nn.Sequential(*layers)
    
    def _get_activation_layer(self, activation: str) -> List[nn.Module]:
        if activation == 'relu':
            return [nn.ReLU()]
        elif activation == 'tanh':
            return [nn.Tanh()]
        elif activation == 'sigmoid':
            return [nn.Sigmoid()]
        elif activation == 'linear':
            return []
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _get_optimizer(self, model: nn.Module, optimizer_config: Dict) -> optim.Optimizer:
        opt_type = optimizer_config['type']
        lr = optimizer_config['config'].get('learning_rate', 0.001)
        if opt_type == 'adam':
            return optim.Adam(model.parameters(), lr=lr)
        elif opt_type == 'sgd':
            return optim.SGD(model.parameters(), lr=lr)
        elif opt_type == 'rmsprop':
            return optim.RMSprop(model.parameters(), lr=lr)
    
    def _get_loss_function(self, loss_type: str) -> nn.Module:
        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_type == 'mse':
            return nn.MSELoss()
        
    def _calculate_loss(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> float:
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y)
        return float(loss.item())
    
    def get_best_architectures(self, metric='accuracy', top_k=5) -> List[Dict]:
        sorted_history = sorted(
            self.history,
            key=lambda x: x['metrics'][metric],
            reverse=True
        )
        return sorted_history[:top_k]

class MLEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, dataset_size: int = 1000):
        super().__init__()
        
        self.layer_types = ['Dense']
        self.activations = ['relu', 'tanh', 'sigmoid', 'linear']
        self.optimizers = ['adam', 'sgd', 'rmsprop']
        
        self.action_space = spaces.Dict({
            'layer_type': spaces.Discrete(len(self.layer_types)), 
            'neurons': spaces.Discrete(512),
            'activation': spaces.Discrete(len(self.activations)),
            'optimizer': spaces.Discrete(len(self.optimizers)),
            'learning_rate': spaces.Box(low=0.0001, high=0.1, shape=(1,), dtype=np.float32)
        })
        
        self.observation_space = spaces.Dict({
            'accuracy': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'loss': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'params': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'training_time': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'convergence': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'f1_score': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        
        self.current_architecture = MLArchitecture()
        self.model_tester = ModelTester()
        self.dataset_size = dataset_size
        self.episode_count = 0
        self.state = None
        self.reset()
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        layer_type = self.layer_types[action['layer_type']]
        activation = self.activations[action['activation']]
        optimizer = self.optimizers[action['optimizer']]
        
        layer_config = {'units': int(action['neurons']), 'activation': activation}
        self.current_architecture.add_layer(layer_type, layer_config)
        
        optimizer_config = {'learning_rate': float(action['learning_rate'])}
        self.current_architecture.set_optimizer(optimizer, optimizer_config)
        self.current_architecture.set_loss('cross_entropy')
        
        metrics = self.model_tester.test_architecture(self.current_architecture)
        
        total_params = sum(layer['config']['units'] for layer in self.current_architecture.layers)
        self.state = {
            'accuracy': np.array([metrics['accuracy']], dtype=np.float32),
            'loss': np.array([1.0 - metrics['accuracy']], dtype=np.float32),
            'params': np.array([total_params], dtype=np.float32),
            'training_time': np.array([metrics['training_time']], dtype=np.float32),
            'convergence': np.array([metrics['convergence_rate']], dtype=np.float32),
            'f1_score': np.array([metrics['f1']], dtype=np.float32)
        }
        
        reward = self._calculate_reward(metrics)
        
        terminated = metrics['accuracy'] > 0.95
        truncated = len(self.current_architecture.layers) >= 10
        
        return self.state, reward, terminated, truncated, metrics
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        accuracy_weight = 1.0
        f1_weight = 0.5
        convergence_weight = 0.3
        
        time_penalty = 0.1
        complexity_penalty = 0.2
        
        accuracy = metrics['accuracy']
        f1 = metrics['f1']
        convergence = metrics['convergence_rate']
        training_time = metrics['training_time']
        num_layers = len(self.current_architecture.layers)
        
        base_reward = (accuracy_weight * accuracy) + (f1_weight * f1)
        convergence_term = convergence_weight * (1.0 / (1.0 + convergence))
        time_term = -time_penalty * (training_time / 10)
        complexity_term = -complexity_penalty * (num_layers / 10)
        
        low_acc_penalty = -0.5 if accuracy < 0.5 else 0
        
        reward = base_reward + convergence_term + time_term + complexity_term + low_acc_penalty
        return reward
    
    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_count += 1
        self.model_tester.update_curriculum(self.episode_count)
        self.current_architecture = MLArchitecture()
        
        self.state = {
            'accuracy': np.array([0.0], dtype=np.float32),
            'loss': np.array([1.0], dtype=np.float32),
            'params': np.array([0.0], dtype=np.float32),
            'training_time': np.array([0.0], dtype=np.float32),
            'convergence': np.array([0.0], dtype=np.float32),
            'f1_score': np.array([0.0], dtype=np.float32)
        }
        return self.state, {}

########################################
# PPO Agent and Trainer
########################################

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

class PPOAgent(nn.Module):
    def __init__(self, state_dim: int, layer_type_dim: int, neurons_dim: int, activation_dim: int, optimizer_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.layer_type_dim = layer_type_dim
        self.neurons_dim = neurons_dim
        self.activation_dim = activation_dim
        self.optimizer_dim = optimizer_dim
        
        hidden_size = 128
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.layer_type_head = nn.Linear(hidden_size, layer_type_dim)
        self.neurons_head = nn.Linear(hidden_size, neurons_dim)
        self.activation_head = nn.Linear(hidden_size, activation_dim)
        self.optimizer_head = nn.Linear(hidden_size, optimizer_dim)
        
        self.lr_mean = nn.Linear(hidden_size, 1)
        self.lr_log_std = nn.Parameter(torch.zeros(1,1))
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def policy(self, state: torch.Tensor):
        features = self.policy_net(state)
        
        lt_logits = self.layer_type_head(features)
        neu_logits = self.neurons_head(features)
        act_logits = self.activation_head(features)
        opt_logits = self.optimizer_head(features)
        
        lr_mean = self.lr_mean(features)
        lr_log_std = self.lr_log_std.expand_as(lr_mean)
        
        return lt_logits, neu_logits, act_logits, opt_logits, lr_mean, lr_log_std
    
    def value(self, state: torch.Tensor):
        return self.value_net(state)
    
    def act(self, state: torch.Tensor):
        with torch.no_grad():
            (lt_logits, neu_logits, act_logits, opt_logits, lr_mean, lr_log_std) = self.policy(state)
            
            lt_dist = torch.distributions.Categorical(logits=lt_logits)
            neu_dist = torch.distributions.Categorical(logits=neu_logits)
            act_dist = torch.distributions.Categorical(logits=act_logits)
            opt_dist = torch.distributions.Categorical(logits=opt_logits)
            
            lt_action = lt_dist.sample()
            neu_action = neu_dist.sample()
            act_action = act_dist.sample()
            opt_action = opt_dist.sample()
            
            lr_std = lr_log_std.exp()
            lr_dist = torch.distributions.Normal(lr_mean, lr_std)
            lr_action = lr_dist.sample()
            
            actions = (lt_action.item(), neu_action.item(), act_action.item(), opt_action.item(), lr_action.item())
            
            log_prob = (lt_dist.log_prob(lt_action) 
                        + neu_dist.log_prob(neu_action) 
                        + act_dist.log_prob(act_action) 
                        + opt_dist.log_prob(opt_action) 
                        + lr_dist.log_prob(lr_action).sum())
            
            value = self.value(state)
        return actions, log_prob, value

def ppo_update(agent, optimizer, states, actions, old_log_probs, returns, advantages, clip_ratio=0.2, epochs=5):
    for _ in range(epochs):
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        
        for start in range(0, len(states), 32):
            end = start + 32
            batch_idx = indices[start:end]
            
            s_batch = states[batch_idx]
            a_batch = actions[batch_idx]
            old_log_batch = old_log_probs[batch_idx]
            ret_batch = returns[batch_idx]
            adv_batch = advantages[batch_idx]
            
            (lt_logits, neu_logits, act_logits, opt_logits, lr_mean, lr_log_std) = agent.policy(s_batch)
            value = agent.value(s_batch).squeeze(-1)
            
            lt_dist = torch.distributions.Categorical(logits=lt_logits)
            neu_dist = torch.distributions.Categorical(logits=neu_logits)
            act_dist = torch.distributions.Categorical(logits=act_logits)
            opt_dist = torch.distributions.Categorical(logits=opt_logits)
            
            lt_action = a_batch[:,0]
            neu_action = a_batch[:,1]
            act_action = a_batch[:,2]
            opt_action = a_batch[:,3]
            lr_action = a_batch[:,4].unsqueeze(-1)
            
            lr_std = lr_log_std.exp()
            lr_dist = torch.distributions.Normal(lr_mean, lr_std)
            
            new_log_probs = (lt_dist.log_prob(lt_action)
                              + neu_dist.log_prob(neu_action)
                              + act_dist.log_prob(act_action)
                              + opt_dist.log_prob(opt_action)
                              + lr_dist.log_prob(lr_action).sum(dim=-1))
            
            ratio = torch.exp(new_log_probs - old_log_batch)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_batch
            
            value_loss = F.mse_loss(value, ret_batch)
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = (lt_dist.entropy()+neu_dist.entropy()+act_dist.entropy()+opt_dist.entropy()+lr_dist.entropy().sum(dim=-1)).mean() * 0.001
            
            loss = policy_loss + 0.5 * value_loss - entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def compute_gae(rewards, values, is_terminals, gamma=0.99, lam=0.95):
    values = values + [0.0]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - is_terminals[step]) - values[step]
        gae = delta + gamma * lam * (1 - is_terminals[step]) * gae
        returns.insert(0, gae + values[step])
    return returns

########################################
# Trainer Class
########################################

class Trainer:
    def __init__(self, max_episodes=1000):
        self.env = MLEnvironment()
        self.max_episodes = max_episodes
        
        # Dimensions
        state_dim = 6 # accuracy, loss, params, training_time, convergence, f1_score
        layer_type_dim = 1
        neurons_dim = 512
        activation_dim = 4
        optimizer_dim = 3
        
        self.agent = PPOAgent(state_dim, layer_type_dim, neurons_dim, activation_dim, optimizer_dim)
        self.ppo_optimizer = optim.Adam(self.agent.parameters(), lr=0.0003)
        
        self.buffer = RolloutBuffer()
        self.episode_count = 0

    def run_episode(self):
        # Run one episode of interaction
        state, _ = self.env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        transitions = []
        
        while not (done or truncated):
            state_array = np.concatenate([state[k] for k in sorted(state.keys())])
            s_tensor = torch.FloatTensor(state_array).unsqueeze(0)
            actions, log_prob, value = self.agent.act(s_tensor)
            
            act_dict = {
                'layer_type': int(actions[0]),
                'neurons': int(actions[1]),
                'activation': int(actions[2]),
                'optimizer': int(actions[3]),
                # scale learning rate
                'learning_rate': 0.0001 + (0.1 - 0.0001)* (1/(1+math.e**(-actions[4])))
            }

            next_state, reward, done, truncated, info = self.env.step(act_dict)
            
            self.buffer.states.append(s_tensor.squeeze(0))
            self.buffer.actions.append(torch.tensor(actions, dtype=torch.float32))
            self.buffer.log_probs.append(log_prob)
            self.buffer.rewards.append(reward)
            self.buffer.is_terminals.append(float(done or truncated))
            self.buffer.values.append(value.item())
            
            total_reward += reward
            state = next_state
        
        # Update PPO
        returns = compute_gae(self.buffer.rewards, self.buffer.values, self.buffer.is_terminals)
        returns = torch.FloatTensor(returns)
        values_t = torch.FloatTensor(self.buffer.values)
        advantages = returns - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.stack(self.buffer.states)
        actions_t = torch.stack(self.buffer.actions)
        old_log_probs_t = torch.stack(self.buffer.log_probs)

        ppo_update(self.agent, self.ppo_optimizer, states_t, actions_t, old_log_probs_t, returns, advantages)

        self.buffer.clear()
        self.episode_count += 1

        # Extract metrics for GUI
        # info dict has final metrics from environment step
        # info is from the last step of the episode, which should have final metrics.
        metrics = {
            'episode': self.episode_count,
            'accuracy': float(state['accuracy'][0]),
            'f1': float(state['f1_score'][0]),
            'reward': float(total_reward)
        }
        return metrics

    def get_best_architectures(self):
        return self.env.model_tester.get_best_architectures()


########################################
# GUI Code
########################################

class TrainingController(QThread):
    metrics_updated = pyqtSignal(dict)
    best_arch_updated = pyqtSignal(list)
    training_stopped = pyqtSignal()
    
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self._paused = False
        self._stopped = False

    def run(self):
        for _ in range(self.trainer.max_episodes):
            if self._stopped:
                break
            while self._paused:
                time.sleep(0.1)
            
            metrics = self.trainer.run_episode()
            best_archs = self.trainer.get_best_architectures()
            
            self.metrics_updated.emit(metrics)
            self.best_arch_updated.emit(best_archs)

            if self._stopped:
                break
        self.training_stopped.emit()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stopped = True

class MainWindow(QWidget):
    def __init__(self, trainer):
        super().__init__()
        self.setWindowTitle("Meta-Model Trainer GUI")
        self.trainer = trainer
        self.controller = TrainingController(self.trainer)
        
        self.controller.metrics_updated.connect(self.update_metrics)
        self.controller.best_arch_updated.connect(self.update_best_arch)
        self.controller.training_stopped.connect(self.on_training_stopped)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.episode_label = QLabel("Episode: 0")
        self.accuracy_label = QLabel("Accuracy: 0.00")
        header_layout = QHBoxLayout()
        header_layout.addWidget(self.episode_label)
        header_layout.addWidget(self.accuracy_label)
        layout.addLayout(header_layout)

        self.plot_widget = pg.PlotWidget(title="Accuracy Over Episodes")
        self.accuracy_curve = self.plot_widget.plot(pen='y')
        self.accuracy_data = []
        
        layout.addWidget(self.plot_widget)

        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        self.insights_text.setPlaceholderText("Insights and emergent behaviors will appear here...")
        layout.addWidget(self.insights_text)

        self.best_arch_table = QTableWidget()
        self.best_arch_table.setColumnCount(3)
        self.best_arch_table.setHorizontalHeaderLabels(["Accuracy", "F1 Score", "Layers"])
        layout.addWidget(self.best_arch_table)

        btn_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")

        self.pause_btn.clicked.connect(self.controller.pause)
        self.resume_btn.clicked.connect(self.controller.resume)
        self.stop_btn.clicked.connect(self.controller.stop)

        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.resume_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def update_metrics(self, metrics):
        episode = metrics.get('episode', 0)
        accuracy = metrics.get('accuracy', 0.0)

        self.episode_label.setText(f"Episode: {episode}")
        self.accuracy_label.setText(f"Accuracy: {accuracy:.3f}")

        self.accuracy_data.append((episode, accuracy))
        x_data = [d[0] for d in self.accuracy_data]
        y_data = [d[1] for d in self.accuracy_data]
        self.accuracy_curve.setData(x_data, y_data)

        if accuracy > 0.9:
            self.insights_text.append(f"Episode {episode}: High accuracy! Emergent behavior shows effective architectures.")
        else:
            self.insights_text.append(f"Episode {episode}: Accuracy {accuracy:.3f} - Learning...")

    def update_best_arch(self, best_archs):
        self.best_arch_table.setRowCount(len(best_archs))
        for i, arch_info in enumerate(best_archs):
            metrics = arch_info['metrics']
            architecture = arch_info['architecture']
            accuracy = metrics.get('accuracy', 0.0)
            f1 = metrics.get('f1', 0.0)
            layers = len(architecture['layers'])

            self.best_arch_table.setItem(i, 0, QTableWidgetItem(f"{accuracy:.3f}"))
            self.best_arch_table.setItem(i, 1, QTableWidgetItem(f"{f1:.3f}"))
            self.best_arch_table.setItem(i, 2, QTableWidgetItem(str(layers)))

    def on_training_stopped(self):
        self.insights_text.append("Training stopped. Check top architectures and patterns discovered.")


def main():
    app = QApplication(sys.argv)
    trainer = Trainer(max_episodes=100)  # Set desired max episodes
    window = MainWindow(trainer)
    window.show()
    window.controller.start()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
