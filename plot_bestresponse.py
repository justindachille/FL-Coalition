import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

def plot_dataframes(exp_dir):
    # Load dataframes from pickle files
    with open(os.path.join(exp_dir, 'price_df.pkl'), 'rb') as f:
        price_df = pickle.load(f)
    with open(os.path.join(exp_dir, 'quality_df.pkl'), 'rb') as f:
        quality_df = pickle.load(f)

    # Plot price history
    price_df.plot()
    plt.title('Price History')
    plt.xlabel('Round')
    plt.ylabel('Price')
    plt.legend(title='Client', loc='best')
    plt.savefig(os.path.join(exp_dir, 'price_history.png'))  # Save figure to exp_dir
    plt.show()

    # Plot quality history
    quality_df.plot()
    plt.title('Quality History')
    plt.xlabel('Round')
    plt.ylabel('Quality')
    plt.legend(title='Client', loc='best')
    plt.savefig(os.path.join(exp_dir, 'quality_history.png'))  # Save figure to exp_dir
    plt.show()

# Substitute 'exp_dir' with your directory path
plot_dataframes('experiment__20230721-205448_e277d8')
