import numpy as np
import gymnasium as gym
from tqdm import tqdm

class Sarsa():
    def __init__(self, env:gym.Env, gamma:float=0.99, epsilon:float=0.1, alpha:float=0.5):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def e_greedy(self, state, epsilon):
        if np.random.rand() < epsilon:
            # Explore - Take a random action from the action space
            return np.random.choice(range(self.env.action_space.n))
        else:
            # Exploit - Follow the policy for the state
            max_indices = [i for i, v in enumerate(self.Q[state]) if v == max(self.Q[state])]
            return np.random.choice(max_indices)
        
    def run(self, episodes:int):
        for episode in tqdm(range(episodes), desc="Episode"):

            state, info = self.env.reset()
            action = self.e_greedy(state, self.epsilon)

            while True:
                next_state, reward, done, truncated, _ = self.env.step(action)

                self.Q[(state, action)] += self.alpha * (reward + self.gamma * self.Q[(next_state, self.e_greedy(next_state, self.epsilon))] - self.Q[(state, action)])

                state = next_state
                action = self.e_greedy(state, self.epsilon) 

                if done or truncated:
                    break

if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    sarsa = Sarsa(env)

    sarsa.run(500)

    print(sarsa.Q)

    env = gym.make("CliffWalking-v0", render_mode='human')
    state, info = env.reset()
    while True:
        env.render()
        action = sarsa.e_greedy(state, epsilon=0)
        next_state, reward, done, info, _ = env.step(action)
        state = next_state  
        if done:
            break