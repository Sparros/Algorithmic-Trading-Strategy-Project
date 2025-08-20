from backtesting import Strategy
import pandas as pd

class MLProbabilisticStrategy(Strategy):
    base_allocation = 0.2   # max fraction of equity per trade
    sl_percentage = 0.03    # stop loss %
    tp_percentage = 0.1     # take profit %
    slippage = 0.0005

    def init(self):
        # Expect dataframe to have columns: 'Prob_Up', 'Prob_Down'
        self.prob_up = self.data['Prob_Up']
        self.prob_down = self.data['Prob_Down']

    def next(self):
        p_up = self.prob_up[-1]
        p_down = self.prob_down[-1]

        current_price = self.data.Close[-1]
        current_value = self.equity
        position_value = self.position.size * current_price

        # Determine position sizing based on confidence
        long_target = current_value * self.base_allocation * p_up
        short_target = current_value * self.base_allocation * p_down

        # Net target: long positive, short negative
        target_position_value = long_target - short_target

        if target_position_value > position_value:
            # Need to buy more
            diff = target_position_value - position_value
            buy_price = current_price * (1 + self.slippage)
            shares = int(diff / buy_price)
            if shares > 0:
                sl_price = current_price * (1 - self.sl_percentage)
                tp_price = current_price * (1 + self.tp_percentage)
                self.buy(size=shares, sl=sl_price, tp=tp_price)

        elif target_position_value < position_value:
            # Need to sell (reduce or flip to short)
            diff = position_value - target_position_value
            sell_price = current_price * (1 - self.slippage)
            shares = int(diff / sell_price)
            if shares > 0:
                sl_price = current_price * (1 + self.sl_percentage)
                tp_price = current_price * (1 - self.tp_percentage)
                self.sell(size=shares, sl=sl_price, tp=tp_price)
