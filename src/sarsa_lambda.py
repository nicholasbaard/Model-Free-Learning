import numpy as np
import gymnasium as gym
from tqdm import tqdm

from utils import plot_heatmap, create_gif

class SarsaLambda():
    def __init__(self, env:gym.Env, gamma:float=0.9, epsilon:float=0.1, alpha:float=0.5, lambda_:float=0.3, state_shape=(4,12)):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.lambda_ = lambda_
        self.state_shape = state_shape

        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def e_greedy(self, state, epsilon):
        if np.random.rand() < epsilon:
            # Explore - Take a random action from the action space
            return np.random.choice(range(self.env.action_space.n))
        else:
            # Exploit - Follow the policy for the state
            max_indices = [i for i, v in enumerate(self.Q[state]) if v == max(self.Q[state])]
            return np.random.choice(max_indices)
        
    def run(self, episodes:int=500):
        for episode in tqdm(range(episodes), desc="Episode"):

            state, info = self.env.reset()
            action = self.e_greedy(state, self.epsilon) 
            e = np.zeros((self.env.observation_space.n, self.env.action_space.n))

            while True:
                
                next_state, reward, done, truncated, _ = self.env.step(action)
                delta = reward + self.gamma * self.Q[(next_state, self.e_greedy(next_state, self.epsilon))] - self.Q[(state, action)]
                e[(state, action)] += 1

                for s, a in np.ndindex(self.Q.shape):

                    self.Q[(s, a)] += self.alpha * delta * e[(s, a)]
                    e[(s, a)] *= self.gamma * self.lambda_

                state = next_state
                action = self.e_greedy(state, self.epsilon) 

                if done or truncated:
                    break
                
            plot_heatmap(self.Q, self.state_shape, episode, save_dir=f"../results/sarsa_lambda_{self.lambda_}")

        create_gif(f"../results/sarsa_lambda_{self.lambda_}", episodes)

if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")

    lambdas = [0.0, 0.3, 0.5, 0.7, 0.9]

    for lambda_ in lambdas:
        print(f"\nRunning with lambda={lambda_}")
        sarsa_l = SarsaLambda(env, lambda_=lambda_)
        sarsa_l.run()

    # print(sarsa_l.Q)

    # env = gym.make("CliffWalking-v0", render_mode='human')
    # state, info = env.reset()
    # while True:
    #     env.render()
    #     action = sarsa_l.e_greedy(state, epsilon=0)
    #     next_state, reward, done, info, _ = env.step(action)
    #     state = next_state  
    #     if done:
    #         break