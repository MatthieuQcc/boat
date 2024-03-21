import numpy as np

class TradingEnv:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.done = False
        self.state = self.get_state()
        return self.state

    def get_state(self):
        return np.array([
            self.data['Open'].iloc[self.current_step],
            self.data['High'].iloc[self.current_step],
            self.data['Low'].iloc[self.current_step],
            self.data['Close'].iloc[self.current_step]
        ])

    def step(self, action):
        if self.done:
            raise Exception("Episode is done")
        reward = 0
        # Execute action: 0 -> hold, 1 -> buy, 2 -> sell
        if action == 1:  # Buy
            reward -= self.data['Close'].iloc[self.current_step]  # Buy at close price
        elif action == 2:  # Sell
            reward += self.data['Close'].iloc[self.current_step]  # Sell at close price
        self.balance += reward
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
        next_state = self.get_state()
        return next_state, reward, self.done, {}
