from freqtrade.strategy import IStrategy
from pandas import DataFrame

class SampleStrategy(IStrategy):
    timeframe = '5m'
    minimal_roi = {"0": 0.02}
    stoploss = -0.02
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count: int = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = dataframe['close'].ewm(span=9, adjust=False).mean()
        dataframe['ema_slow'] = dataframe['close'].ewm(span=21, adjust=False).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['close'] > dataframe['ema_fast']), 'enter_long'] = 1
        dataframe.loc[
            (dataframe['ema_fast'] < dataframe['ema_slow']) &
            (dataframe['close'] < dataframe['ema_fast']), 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        return dataframe
