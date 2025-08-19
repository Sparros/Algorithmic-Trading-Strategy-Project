from backtesting import Strategy
import pandas as pd

class ContinuousAllocationStrategy(Strategy):
    allocation_percentage = 0.5
    sl_percentage = 0.05
    slippage = 0.0005

    def init(self):
        # Store the prediction signal
        self.signal = self.data['Prediction']

    def next(self):
        prediction = self.signal[-1]
        current_account_value = self.equity
        current_position_value = self.position.size * self.data.Close[-1]

        if prediction == 1:
            target_position_value = current_account_value * self.allocation_percentage
        else:
            target_position_value = 0

        if target_position_value > current_position_value:
            amount_to_buy = target_position_value - current_position_value
            buy_price = self.data.Close[-1] * (1 + self.slippage)
            shares_to_buy = int(amount_to_buy / buy_price)
            if shares_to_buy > 0:
                # Add trailing stop (fixed % below entry)
                sl_price = self.data.Close[-1] * (1 - self.sl_percentage)
                self.buy(size=shares_to_buy, sl=sl_price)

        elif target_position_value < current_position_value:
            amount_to_sell = current_position_value - target_position_value
            sell_price = self.data.Close[-1] * (1 - self.slippage)
            shares_to_sell = int(amount_to_sell / sell_price)
            if shares_to_sell > 0:
                self.sell(size=shares_to_sell)
