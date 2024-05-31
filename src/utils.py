import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Function to plot the heatmap
def plot_heatmap(Q, state_shape, episode, save_dir):
    # Compute the value function (max Q-value for each state)
    # check if folder exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    V = np.zeros(state_shape[0]*state_shape[1])
    for state in range(Q.shape[0]):
        V[state] = max(Q[state])
    
    # Reshape the value function to the shape of the state space
    V = V.reshape(state_shape)
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(V, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(f'Value Function Heatmap - Episode {episode}')
    plt.xlabel('State Dimension 1')
    plt.ylabel('State Dimension 2')
    
    # Save the plot as an image
    plt.savefig(f'{save_dir}/heatmap_{episode}.png')
    plt.close()

def delete_image_files(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.endswith('.png'):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                delete_image_files(file_path)

def create_gif(save_dir, num_episodes):
    # Create a list of image file paths
    image_files = [f'{save_dir}/heatmap_{episode}.png' for episode in range(num_episodes)]

    # Create an animated GIF
    with imageio.get_writer(f'{save_dir}/value_function_animation.gif', mode='I', duration=0.01) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    delete_image_files(save_dir)
    