import gym
import numpy as np
import cv2

class SnakeGame:
    def __init__(self, grid_size=10, cell_size=20):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(grid_size * grid_size,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]
        self.food = self._place_food()
        self.done = False
        return self._get_observation()

    def _place_food(self):
        while True:
            food = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if food not in self.snake:
                return food

    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for x, y in self.snake:
            obs[x, y] = 1
        fx, fy = self.food
        obs[fx, fy] = 2
        return obs.flatten()

    def step(self, action):
        head_x, head_y = self.snake[0]
        if action == 0:
            head_x -= 1
        elif action == 1:
            head_x += 1
        elif action == 2:
            head_y -= 1
        elif action == 3:
            head_y += 1

        new_head = (head_x, head_y)
        if (head_x < 0 or head_x >= self.grid_size or
            head_y < 0 or head_y >= self.grid_size or
            new_head in self.snake):
            self.done = True
            reward = -1
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.food = self._place_food()
                reward = 1
            else:
                self.snake.pop()
                reward = 0
        return self._get_observation(), reward, self.done, {}

    def render(self):
        img = np.zeros((self.grid_size * self.cell_size, self.grid_size * self.cell_size, 3), dtype=np.uint8)
        for x, y in self.snake:
            cv2.rectangle(img,
                         (y * self.cell_size, x * self.cell_size),
                         ((y + 1) * self.cell_size, (x + 1) * self.cell_size),
                         (0, 255, 0), -1)
        fx, fy = self.food
        cv2.rectangle(img,
                     (fy * self.cell_size, fx * self.cell_size),
                     ((fy + 1) * self.cell_size, (fx + 1) * self.cell_size),
                     (0, 0, 255), -1)
        return img

class PuckWorldGame:
    def __init__(self, size=200):
        self.position = np.array([0.5, 0.5])
        self.target = np.array([0.5, 0.5])
        self.velocity = np.array([0.0, 0.0])
        self.dt = 0.1
        self.size = size
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.position = np.random.rand(2)
        self.target = np.random.rand(2)
        self.velocity = np.zeros(2)
        return self._get_observation()

    def _get_observation(self):
        return np.concatenate([self.position, self.velocity, self.target]).astype(np.float32)

    def step(self, action):
        force = np.zeros(2)
        if action == 0:
            force[1] = -0.1
        elif action == 1:
            force[1] = 0.1
        elif action == 2:
            force[0] = -0.1
        elif action == 3:
            force[0] = 0.1

        self.velocity += self.dt * force
        self.position += self.dt * self.velocity
        self.position = np.clip(self.position, 0, 1)
        self.velocity *= 0.9
        dist = np.linalg.norm(self.position - self.target)
        reward = -dist
        done = dist < 0.05
        if done:
            reward += 1
        return self._get_observation(), reward, done, {}

    def render(self):
        img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255
        px, py = (self.position * self.size).astype(int)
        cv2.circle(img, (px, py), 10, (255, 0, 0), -1)
        tx, ty = (self.target * self.size).astype(int)
        cv2.circle(img, (tx, ty), 5, (0, 0, 255), -1)
        return img

class PongEnv:
    def __init__(self):
        self.env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(80, 80, 1), dtype=np.float32)

    def reset(self):
        obs, _ = self.env.reset()
        return self._preprocess(obs)

    def step(self, action):
        atari_action = [0, 2, 3][action]
        obs, reward, done, truncated, info = self.env.step(atari_action)
        return self._preprocess(obs), reward, done or truncated, info

    def _preprocess(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (80, 80))
        obs = obs / 255.0
        return obs[:, :, np.newaxis].astype(np.float32)

    def render(self):
        return self.env.render()