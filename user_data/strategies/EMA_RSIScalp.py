from freqtrade.strategy import IStrategy
from pandas import DataFrame

class EMA_RSIScalp(IStrategy):
    timeframe = '5m'
    minimal_roi = {"0": 0.008}  # ~0.8%
    stoploss = -0.007             # -0.7% slightly wider to allow entries
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = dataframe['close'].ewm(span=9, adjust=False).mean()
        dataframe['ema_slow'] = dataframe['close'].ewm(span=21, adjust=False).mean()
        dataframe['rsi'] = dataframe['close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / (
            dataframe['close'].diff().abs().ewm(alpha=1/14).mean() + 1e-9
        ) * 100
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['rsi'] > 48) &
            (dataframe['close'] > dataframe['ema_fast']),
            'enter_long'
        ] = 1

        dataframe.loc[
            (dataframe['ema_fast'] < dataframe['ema_slow']) &
            (dataframe['rsi'] < 52) &
            (dataframe['close'] < dataframe['ema_fast']),
            'enter_short'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
