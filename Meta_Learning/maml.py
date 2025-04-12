import torch
import torch.optim as optim
import numpy as np
from model import PolicyNetwork
import os
from datetime import datetime
import imageio
import matplotlib.pyplot as plt

class MAMLAgent:
    def __init__(self, envs, config, output_dir):
        self.envs = envs
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        for env_name, env in envs.items():
            input_shape = env.observation_space.shape
            output_dim = env.action_space.n
            self.models[env_name] = PolicyNetwork(
                input_shape, output_dim, env_name, config["hidden_dims"], config["conv_filters"]
            ).to(self.device)
        self.meta_optimizer = optim.Adam(
            [param for model in self.models.values() for param in model.parameters()],
            lr=config["meta_lr"]
        )
        self.inner_lr = config["inner_lr"]
        self.game_rewards = {name: [] for name in envs.keys()}

    def inner_loop(self, env, env_name, params, num_steps):
        temp_model = PolicyNetwork(
            env.observation_space.shape, env.action_space.n, env_name,
            self.config["hidden_dims"], self.config["conv_filters"]
        ).to(self.device)
        temp_model.load_state_dict(params)
        optimizer = optim.SGD(temp_model.parameters(), lr=self.inner_lr)

        for step in range(num_steps):
            obs = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
            done = False
            total_loss = 0
            total_reward = 0
            episode_steps = 0

            while not done:
                probs = temp_model(obs)
                action = torch.multinomial(probs, 1).item()
                next_obs, reward, done, _ = env.step(action)
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                loss = -reward * torch.log(probs[0, action] + 1e-10)
                total_loss += loss
                total_reward += reward
                episode_steps += 1
                obs = next_obs

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % self.config["log_interval"] == 0:
                print(f"Game: {env_name}, Inner Step: {step}, Episode Reward: {total_reward:.4f}, Steps: {episode_steps}")

        return temp_model.state_dict()

    def train(self, num_iterations, k_shots):
        losses = []
        for iteration in range(num_iterations):
            meta_loss = 0
            for env_name, env in self.envs.items():
                params = {k: v.clone() for k, v in self.models[env_name].state_dict().items()}
                adapted_params = self.inner_loop(env, env_name, params, num_steps=k_shots)

                temp_model = PolicyNetwork(
                    env.observation_space.shape, env.action_space.n, env_name,
                    self.config["hidden_dims"], self.config["conv_filters"]
                ).to(self.device)
                temp_model.load_state_dict(adapted_params)
                obs = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                done = False
                task_loss = 0
                total_reward = 0
                episode_steps = 0

                while not done:
                    probs = temp_model(obs)
                    action = torch.multinomial(probs, 1).item()
                    next_obs, reward, done, _ = env.step(action)
                    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                    task_loss -= reward * torch.log(probs[0, action] + 1e-10)
                    total_reward += reward
                    episode_steps += 1
                    obs = next_obs

                meta_loss += task_loss
                self.game_rewards[env_name].append(total_reward)

                if iteration % self.config["log_interval"] == 0:
                    print(f"Iteration {iteration}, Game: {env_name}, Meta Episode Reward: {total_reward:.4f}, Steps: {episode_steps}")

            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

            losses.append(meta_loss.item())
            if iteration % self.config["log_interval"] == 0:
                print(f"Iteration {iteration}, Meta Loss: {meta_loss.item():.4f}")

            if iteration % 20 == 0:
                self.save_model(iteration)

        return losses

    def save_model(self, iteration):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        for env_name, model in self.models.items():
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, f"{env_name}_model_iter{iteration}_{timestamp}.pth")
            )
        print(f"Models saved at iteration {iteration}")

    def evaluate_and_record(self, env, env_name, num_episodes):
        video_dir = os.path.join(self.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        rewards = []
        video_frames = []
        for episode in range(num_episodes):
            params = {k: v.clone() for k, v in self.models[env_name].state_dict().items()}
            adapted_params = self.inner_loop(env, env_name, params, num_steps=self.config["k_shots"])
            temp_model = PolicyNetwork(
                env.observation_space.shape, env.action_space.n, env_name,
                self.config["hidden_dims"], self.config["conv_filters"]
            ).to(self.device)
            temp_model.load_state_dict(adapted_params)

            obs = env.reset()
            total_reward = 0
            done = False
            frames = []

            while not done:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                probs = temp_model(obs_tensor)
                action = torch.multinomial(probs, 1).item()
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            rewards.append(total_reward)
            if episode == 0:
                video_frames = frames

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(video_dir, f"{env_name}_eval_{timestamp}.mp4")
        with imageio.get_writer(video_path, fps=30, codec='libx264') as writer:
            for frame in video_frames:
                writer.append_data(frame)
        print(f"Video saved for {env_name} at {video_path}")

        return np.mean(rewards)

    def save_plots(self, losses):
        plot_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Meta Loss")
        plt.title("Meta-Learning Loss Across Snake, PuckWorld, Pong")
        plt.savefig(os.path.join(plot_dir, "training_loss.png"))
        plt.close()

        plt.figure(figsize=(15, 5))
        for env_name in self.envs.keys():
            plt.plot(self.game_rewards[env_name], label=env_name)
        plt.xlabel("Iteration")
        plt.ylabel("Meta Episode Reward")
        plt.title("Per-Game Rewards During Meta-Training")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "game_rewards.png"))
        plt.close()