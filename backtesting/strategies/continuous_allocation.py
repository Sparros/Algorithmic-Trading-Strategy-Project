from backtesting import Strategy

class ContinuousAllocationStrategy(Strategy):
    allocation_percentage = 0.5
    _slippage = 0.0005  

    def init(self):
        self.signal = self.data.Prediction

    def next(self):
        prediction = self.signal[-1]
        current_account_value = self.equity
        current_position_value = self.position.size * self.data.Close[-1]

        # Determine target position value
        if prediction == 1:
            target_position_value = current_account_value * self.allocation_percentage
        else:
            target_position_value = 0

        # Buy logic
        if target_position_value > current_position_value:
            amount_to_buy = target_position_value - current_position_value
            buy_price = self.data.Close[-1] * (1 + self._slippage)
            shares_to_buy = int(amount_to_buy / buy_price)
            if shares_to_buy > 0:
                self.buy(size=shares_to_buy)

        # Sell logic
        elif target_position_value < current_position_value:
            amount_to_sell = current_position_value - target_position_value
            sell_price = self.data.Close[-1] * (1 - self._slippage)
            shares_to_sell = int(amount_to_sell / sell_price)
            if shares_to_sell > 0:
                self.sell(size=shares_to_sell)