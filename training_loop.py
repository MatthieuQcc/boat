import pandas as pd
import matplotlib.pyplot as plt
import TradingEnv
from network import DQNAgent

num_episodes = 10

# We load all our data
df = pd.read_csv("data/CryptoHourlyData.csv")
mapping = pd.read_csv("data/coins.csv")


# Create a dictionary mapping currency IDs to their names
currency_mapping = mapping.set_index('id')['name'].to_dict()


# Create separate dataframes for each currency
currency_dfs = {}
for currency_id, currency_name in currency_mapping.items():
    currency_df = df[df['ID'] == currency_id].reset_index(drop=True)
    currency_dfs[currency_name] = currency_df



# Define the environment and agent
env = TradingEnv(data=currency_dfs, initial_balance=10000)
agent = DQNAgent(input_size=4, output_size=3)

# Train the agent
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.train()
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
