from backtesting import Strategy
import pandas as pd

class ContinuousAllocationStrategy(Strategy):
    """
    A trading strategy that adjusts its position continuously
    based on a model's prediction.
    
    This strategy assumes the model's prediction (0 or 1) is available
    in a data series named 'Prediction'.
    """
    
    # Define parameters that can be optimized by the Backtest framework
    allocation_percentage = 0.5  # The percentage of equity to be in the stock when the signal is 1
    sl_percentage = 0.05         # The percentage below entry price for a stop-loss order
    slippage = 0.0005            # A small slippage factor to simulate real-world trading costs

    def init(self):
        """
        Initializes the strategy. This method is called once at the start.
        """
        # Store the prediction signal from the data.
        # This is a key step to link the model's output to the strategy logic.
        self.signal = self.data['Prediction']

    def next(self):
        """
        This method is called on each new bar (e.g., a new day).
        It contains the core trading logic.
        """
        # Get the prediction for the current bar
        prediction = self.signal[-1]
        
        # Calculate the current account value and position value
        current_account_value = self.equity
        current_position_value = self.position.size * self.data.Close[-1]

        if prediction == 1:
            # If the model predicts a rise, the target position is the defined allocation
            target_position_value = current_account_value * self.allocation_percentage
        else:
            # If the model predicts a non-rise, the target position is 0 (move to cash)
            target_position_value = 0

        # --- Trading Logic to Adjust to Target Position ---
        if target_position_value > current_position_value:
            # We need to buy to increase our position
            amount_to_buy = target_position_value - current_position_value
            buy_price = self.data.Close[-1] * (1 + self.slippage)
            
            # Calculate the number of shares to buy, ensuring we buy at least 1
            shares_to_buy = int(amount_to_buy / buy_price)
            
            if shares_to_buy > 0:
                # Place a buy order and a corresponding trailing stop-loss
                sl_price = self.data.Close[-1] * (1 - self.sl_percentage)
                self.buy(size=shares_to_buy, sl=sl_price)

        elif target_position_value < current_position_value:
            # We need to sell to decrease our position
            amount_to_sell = current_position_value - target_position_value
            sell_price = self.data.Close[-1] * (1 - self.slippage)
            
            # Calculate the number of shares to sell, ensuring we sell at least 1
            shares_to_sell = int(amount_to_sell / sell_price)
            
            # Ensure we don't sell more than we own
            if shares_to_sell > 0 and self.position.size >= shares_to_sell:
                self.sell(size=shares_to_sell)
