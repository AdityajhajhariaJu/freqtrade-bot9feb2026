
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class QuickScalp3m(IStrategy):
    timeframe = '3m'
    minimal_roi = {"0": 0.002}  # ~0.2%
    stoploss = -0.01             # -1%
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = dataframe['close'].ewm(span=8, adjust=False).mean()
        dataframe['ema_slow'] = dataframe['close'].ewm(span=21, adjust=False).mean()
        # RSI (EMA-style)
        up = dataframe['close'].diff().clip(lower=0).ewm(alpha=1/14).mean()
        down = (-dataframe['close'].diff().clip(upper=0)).ewm(alpha=1/14).mean() + 1e-9
        dataframe['rsi'] = up / (up + down) * 100
        dataframe['hh_5'] = dataframe['high'].rolling(5).max()
        dataframe['ll_5'] = dataframe['low'].rolling(5).min()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Fast triggers with small guardrails
        dataframe.loc[
            (
                (dataframe['ema_fast'] > dataframe['ema_slow']) &
                (dataframe['rsi'] > 50) &
                ((dataframe['close'] > dataframe['ema_fast']) | (dataframe['close'] > dataframe['hh_5']))
            ), 'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['ema_fast'] < dataframe['ema_slow']) &
                (dataframe['rsi'] < 50) &
                ((dataframe['close'] < dataframe['ema_fast']) | (dataframe['close'] < dataframe['ll_5']))
            ), 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
