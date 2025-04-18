import torch
import torch.optim as optim
import numpy as np
import psutil
import logging
import os
from datetime import datetime
import csv
from model import PolicyNetwork

logger = logging.getLogger(__name__)

class MAMLAgent:
    def __init__(self, envs, config, save_dir, checkpoint_dir):
        self.envs = envs
        self.config = config
        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.models = {}
        for env_name, env in envs.items():
            input_shape = env.observation_space.shape
            output_dim = env.action_space.n
            logger.info(f"Initializing model for {env_name} with input_shape={input_shape}, output_dim={output_dim}")
            model = PolicyNetwork(
                input_shape, output_dim, env_name, config["hidden_dims"],
                config["conv_filters"], frame_stack=config["frame_stack"], dropout_rate=config["dropout_rate"]
            ).to(self.device)
            self.models[env_name] = model
        self.meta_optimizer = optim.Adam(
            [param for model in self.models.values() for param in model.parameters()],
            lr=config["meta_lr"]
        )
        self.inner_lr = config["inner_lr"]
        self.game_rewards = {name: [] for name in envs.keys()}
        self.checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pth")
        self.csv_file = os.path.join(save_dir, 'training_metrics.csv')
        self.csv_columns = ['iteration', 'meta_loss', 'snake_reward', 'puckworld_reward', 'pong_reward', 'epsilon', 'avg_steps', 'grad_norm', 'entropy', 'memory_mb']
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()

    def get_memory_usage(self):
        process = psutil.Process()
        mem = process.memory_info().rss / (1024 ** 2)  # MB
        return mem

    def compute_entropy(self, probs):
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()

    def get_epsilon(self, iteration):
        epsilon = self.config["epsilon_end"] + (self.config["epsilon_start"] - self.config["epsilon_end"]) * \
                  (selfავಗან염 self.config["epsilon_decay"] ** iteration)
        return epsilon

    def compute_grad_norm(self, parameters):
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def inner_loop(self, env, env_name, model, num_steps, iteration):
        optimizer = optim.Adam(model.parameters(), lr=self.inner_lr)
        total_loss = 0
        total_reward = 0
        episode_count = 0
        epsilon = self.get_epsilon(iteration)
        total_steps = 0
        last_entropy = 0

        for inner_step in range(num_steps):
            obs = env.reset()
            if env_name == "pong":
                obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            else:
                obs = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
            done = False
            episode_loss = 0
            episode_rewards = []
            log_probs = []
            steps = 0

            while not done and steps < self.config["max_steps_per_episode"]:
                probs = model(obs)
                if np.random.rand() < epsilon:
                    action = np.random.randint(env.action_space.n)
                else:
                    action = torch.multinomial(probs, 1).item()
                log_prob = torch.log(probs[0, action] + 1e-10)
                next_obs, reward, done, _ = env.step(action)
                if env_name == "pong":
                    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
                else:
                    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                episode_rewards.append(reward)
                log_probs.append(log_prob)
                steps += 1
                obs = next_obs

            discounted_rewards = []
            R = 0
            for r in episode_rewards[::-1]:
                R = r + self.config["discount_factor"] * R
                discounted_rewards.insert(0, R)
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)

            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-10)

            policy_loss = 0
            for log_prob, R in zip(log_probs, discounted_rewards):
                policy_loss -= log_prob * R

            entropy = self.compute_entropy(probs)
            last_entropy = entropy.item()
            episode_loss = policy_loss - self.config["entropy_bonus"] * entropy

            total_loss += episode_loss
            total_reward += sum(episode_rewards)
            episode_count += 1
            total_steps += steps

            if episode_count % self.config["batch_size"] == 0 or inner_step == num_steps - 1:
                optimizer.zero_grad()
                avg_loss = total_loss / episode_count if episode_count > 0 else total_loss
                logger.debug(f"Inner Loop - Game: {env_name}, Step: {inner_step}, Loss: {avg_loss.item():.4f}, Avg Reward: {(total_reward / episode_count):.4f}, Steps: {steps}, Entropy: {entropy.item():.4f}, Epsilon: {epsilon:.4f}")
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["grad_clip"])
                optimizer.step()
                total_loss = 0
                episode_count = 0

            logger.debug(f"Game: {env_name}, Inner Step: {inner_step}, Episode Reward: {sum(episode_rewards):.4f}, Steps: {steps}, Entropy: {entropy.item():.4f}, Memory: {self.get_memory_usage():.2f} MB")

        avg_steps = total_steps / num_steps
        return model.state_dict(), avg_steps, last_entropy

    def train(self, num_iterations, k_shots):
        losses = []
        logger.info(f"Starting training for {num_iterations} iterations with k_shots={k_shots}")

        for iteration in range(num_iterations):
            meta_loss = 0
            avg_steps_per_game = {}
            logger.info(f"Starting Iteration {iteration}/{num_iterations}")
            epsilon = self.get_epsilon(iteration)

            for env_name, env in self.envs.items():
                logger.debug(f"Training on environment: {env_name}")
                model = self.models[env_name]
                params = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(params)
                adapted_params, avg_steps, entropy = self.inner_loop(env, env_name, model, num_steps=k_shots, iteration=iteration)
                avg_steps_per_game[env_name] = avg_steps

                temp_model = PolicyNetwork(
                    env.observation_space.shape, env.action_space.n, env_name,
                    self.config["hidden_dims"], self.config["conv_filters"],
                    frame_stack=self.config["frame_stack"], dropout_rate=self.config["dropout_rate"]
                ).to(self.device)
                temp_model.load_state_dict(adapted_params)
                obs = env.reset()
                if env_name == "pong":
                    obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
                else:
                    obs = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                done = False
                task_loss = 0
                total_reward = 0
                episode_steps = 0
                episode_rewards = []
                log_probs = []

                while not done and episode_steps < self.config["max_steps_per_episode"]:
                    probs = temp_model(obs)
                    if np.random.rand() < epsilon:
                        action = np.random.randint(env.action_space.n)
                    else:
                        action = torch.multinomial(probs, 1).item()
                    log_prob = torch.log(probs[0, action] + 1e-10)
                    next_obs, reward, done, _ = env.step(action)
                    if env_name == "pong":
                        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
                    else:
                        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                    episode_rewards.append(reward)
                    log_probs.append(log_prob)
                    total_reward += reward
                    episode_steps += 1
                    obs = next_obs

                discounted_rewards = []
                R = 0
                for r in episode_rewards[::-1]:
                    R = r + self.config["discount_factor"] * R
                    discounted_rewards.insert(0, R)
                discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)

                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-10)

                for log_prob, R in zip(log_probs, discounted_rewards):
                    task_loss -= log_prob * R

                meta_loss += task_loss
                self.game_rewards[env_name].append(total_reward)

                logger.debug(f"Iteration {iteration}, Game: {env_name}, Meta Episode Reward: {total_reward:.4f}, Steps: {episode_steps}, Task Loss: {task_loss.item():.4f}, Epsilon: {epsilon:.4f}, Memory: {self.get_memory_usage():.2f} MB")

            self.meta_optimizer.zero_grad()
            logger.debug(f"Iteration {iteration}, Meta Loss Before Backward: {meta_loss.item():.4f}")
            meta_loss.backward()
            grad_norm = self.compute_grad_norm([param for model in self.models.values() for param in model.parameters()])
            torch.nn.utils.clip_grad_norm_([param for model in self.models.values() for param in model.parameters()], self.config["grad_clip"])
            self.meta_optimizer.step()

            losses.append(meta_loss.item())
            avg_steps = sum(avg_steps_per_game.values()) / len(avg_steps_per_game)
            logger.info(f"Iteration {iteration}/{num_iterations}, Meta Loss: {meta_loss.item():.4f}, Avg Steps: {avg_steps:.2f}, Grad Norm: {grad_norm:.4f}")

            # Log to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writerow({
                    'iteration': iteration,
                    'meta_loss': meta_loss.item(),
                    'snake_reward': self.game_rewards['snake'][-1] if self.game_rewards['snake'] else 0,
                    'puckworld_reward': self.game_rewards['puckworld'][-1] if self.game_rewards['puckworld'] else 0,
                    'pong_reward': self.game_rewards['pong'][-1] if self.game_rewards['pong'] else 0,
                    'epsilon': epsilon,
                    'avg_steps': avg_steps,
                    'grad_norm': grad_norm,
                    'entropy': entropy,
                    'memory_mb': self.get_memory_usage()
                })

            if iteration % self.config["log_interval"] == 0:
                self.save_model(iteration)
                self.save_checkpoint(iteration, losses)

            torch.cuda.empty_cache()

        logger.info(f"Training completed after {len(losses)} iterations")
        return losses

    def save_model(self, iteration):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for env_name, model in self.models.items():
            torch.save(
                model.state_dict(),
                os.path.join(self.save_dir, f"{env_name}_model_iter{iteration}_{timestamp}.pth")
            )
        logger.info(f"Models saved at iteration {iteration}")

    def save_checkpoint(self, iteration, losses):
        checkpoint = {
            'iteration': iteration,
            'models': {name: model.state_dict() for name, model in self.models.items()},
            'optimizer': self.meta_optimizer.state_dict(),
            'losses': losses,
            'game_rewards': self.game_rewards
        }
        torch.save(checkpoint, self.checkpoint_file)
        logger.info(f"Checkpoint saved at iteration {iteration}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file, weights_only=False)
            for name, state_dict in checkpoint['models'].items():
                self.models[name].load_state_dict(state_dict)
            self.meta_optimizer.load_state_dict(checkpoint['optimizer'])
            self.game_rewards = checkpoint['game_rewards']
            logger.info(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
            return checkpoint['iteration'], checkpoint['losses']
        return 0, []