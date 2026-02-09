from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame

class RangeReenter5m(IStrategy):
    timeframe = '5m'
    informative_timeframe = '4h'
    minimal_roi = {"0": 10}   # effectively ignored; exits via custom_exit
    stoploss = -0.99           # overridden by custom_stoploss
    process_only_new_candles = True
    startup_candle_count = 240
    use_exit_signal = False
    exit_profit_only = False

    def informative_pairs(self):
        # Build informative pairs for all whitelist pairs
        return [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        informative['range_high'] = informative['high']
        informative['range_low'] = informative['low']
        informative = informative[['date', 'range_high', 'range_low']]

        df = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)
        df['range_high'] = df[f'range_high_{self.informative_timeframe}']
        df['range_low'] = df[f'range_low_{self.informative_timeframe}']
        df['prev_close'] = df['close'].shift(1)
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long: re-enter from below range low
        dataframe.loc[
            (
                (dataframe['prev_close'] < dataframe['range_low']) &
                (dataframe['close'] > dataframe['range_low']) &
                (dataframe['range_low'].notnull())
            ), 'enter_long'] = 1

        # Short: re-enter from above range high
        dataframe.loc[
            (
                (dataframe['prev_close'] > dataframe['range_high']) &
                (dataframe['close'] < dataframe['range_high']) &
                (dataframe['range_high'].notnull())
            ), 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        row = df.loc[df['date'] <= current_time].iloc[-1]
        rh, rl = row['range_high'], row['range_low']
        if trade.is_short:
            sl_price = rh
            return (trade.open_rate - sl_price) / trade.open_rate  # negative
        else:
            sl_price = rl
            return (sl_price - trade.open_rate) / trade.open_rate  # negative

    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        row = df.loc[df['date'] <= current_time].iloc[-1]
        rh, rl = row['range_high'], row['range_low']
        if trade.is_short:
            target = max(rl, trade.open_rate - 2 * (rh - trade.open_rate))
            if current_rate <= target:
                return 'take_profit'
        else:
            target = min(rh, trade.open_rate + 2 * (trade.open_rate - rl))
            if current_rate >= target:
                return 'take_profit'
        return None
