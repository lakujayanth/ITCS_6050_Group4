import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import logging
from datetime import datetime
from model import PolicyNetwork

logger = logging.getLogger(__name__)

def evaluate_and_record(agent, env, env_name, num_episodes, video_dir):
    rewards = []
    video_frames = []
    for episode in range(num_episodes):
        model = agent.models[env_name]
        params = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(params)
        adapted_params, _, _ = agent.inner_loop(env, env_name, model, num_steps=agent.config["k_shots"], iteration=0)

        temp_model = PolicyNetwork(
            env.observation_space.shape, env.action_space.n, env_name,
            agent.config["hidden_dims"], agent.config["conv_filters"],
            frame_stack=agent.config["frame_stack"], dropout_rate=agent.config["dropout_rate"]
        ).to(agent.device)
        temp_model.load_state_dict(adapted_params)

        obs = env.reset()
        total_reward = 0
        done = False
        frames = []
        steps = 0

        while not done and steps < agent.config["max_steps_per_episode"]:
            if episode == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
            if env_name != "pong":
                obs_tensor = obs_tensor.unsqueeze(0)
            probs = temp_model(obs_tensor)
            action = torch.multinomial(probs, 1).item()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        if episode == 0:
            video_frames = frames

        logger.info(f"Evaluation Episode {episode}, Game: {env_name}, Reward: {total_reward:.4f}, Steps: {steps}")

    if video_frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(video_dir, f"{env_name}_eval_{timestamp}.mp4")
        with imageio.get_writer(video_path, fps=30, codec='libx264') as writer:
            for frame in video_frames:
                writer.append_data(frame)
        logger.info(f"Video saved for {env_name} at {video_path}")

    return np.mean(rewards)

def plot_training_results(losses, game_rewards, envs, plot_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot meta-learning loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Meta Loss")
    plt.title("Meta-Learning Loss Across Snake, PuckWorld, Pong")
    loss_plot_path = os.path.join(plot_dir, f"training_loss_{timestamp}.png")
    plt.savefig(loss_plot_path)
    plt.close()
    logger.info(f"Loss plot saved at {loss_plot_path}")

    # Plot per-game rewards
    plt.figure(figsize=(15, 5))
    for env_name in envs.keys():
        plt.plot(game_rewards[env_name], label=env_name)
    plt.xlabel("Iteration")
    plt.ylabel("Meta Episode Reward")
    plt.title("Per-Game Rewards During Meta-Training")
    plt.legend()
    rewards_plot_path = os.path.join(plot_dir, f"game_rewards_{timestamp}.png")
    plt.savefig(rewards_plot_path)
    plt.close()
    logger.info(f"Rewards plot saved at {rewards_plot_path}")