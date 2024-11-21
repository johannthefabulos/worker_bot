import pandas as pd


# Add features to data
def add_features(data, features_list):
    if type(features_list) is not list:
        features_list = [features_list]
    for feature in features_list:
        data = feature.add_feature(data)
    return data

class Feature:
    def __init__(self, name, label=None):
        self.name = name            # ID name of the feature
        self.base_features = []     # All required feature names
        if label:                   # Label: The actual displayed name of the feature
            self.label = label
        else:
            self.label = self.name
    def add_feature(self, data):
        return data

class Period_Feature(Feature):
    def __init__(self, name, periods, label=None):
        if type(periods) is int:
            self.periods = [periods]
        else:
            self.periods = periods
        if not label:
            label = [name + '_' + str(period) for period in self.periods]
        else:
            if type(label) is str and len(self.periods) == 1:
                label = [label]
            elif type(label) is str:
                raise Exception('Single label cannot describe multiple periods')
            elif len(self.periods) != len(label):
                raise Exception('Given ' + str(len(self.periods)) + ' periods and ' + str(len(label)) + ' labels for those periods (must be the same)')
        super().__init__(name, label)

class RSI(Period_Feature):
    def __init__(self, periods=20, label=None):
        super().__init__('RSI', periods, label)

    def add_feature(self, data):
        for i in range(len(self.periods)):
            period = self.periods[i]
            label = self.label[i]
            i = 0
            up = [0]
            down = [0]
            size = len(data.index)
            for i in range(size - 1):
                change = data.loc[i + 1, 'close'] - data.loc[i, 'close']
                up.append(change if change > 0 else 0)
                down.append(-change if change < 0 else 0)
            up = pd.Series(up)
            down = pd.Series(down)
            up = pd.Series(up.ewm(span=period, min_periods=period).mean())
            down = pd.Series(down.ewm(span=period, min_periods=period).mean())
            RSI = pd.Series(up / (up + down), name=label)
            data = data.join(RSI)
        return data

class SMA(Period_Feature):
    def __init__(self, periods=20, label=None):
        super().__init__('SMA', periods, label)

    def add_feature(self, data):
        for i, period in enumerate(self.periods):
            label = self.label[i]
            averages = pd.Series(data['close'].rolling(window=period).mean(), name = label)
            data = data.join(averages)
        return data

class EMA(Period_Feature):
    def __init__(self, periods=20, label=None):
        super().__init__('EMA', periods, label)

    def add_feature(self, data):
        for i, period in enumerate(self.periods):
            label = self.label[i]
            averages = pd.Series(data['close'].ewm(span=period, min_periods=period).mean(), name=label)
            data = data.join(averages)
        return data

class momentum(Period_Feature):
    def __init__(self, periods=20, label=None):
        super().__init__('Momentum', periods, label)

    def add_feature(self, data):
        for i, period in enumerate(self.periods):
            label = self.label[i]
            momentum = pd.Series(data['close'].diff(period), name=label)
            data = data.join(momentum)
        return data

class bollinger(Feature):
    def __init__(self, periods=20):
        if type(periods) is int:
            periods = [periods]
        labels = []
        for period in periods:
            labels.append('Bollinger_T_' + str(period))
            labels.append('Bollinger_B_' + str(period))
        self.periods = periods

        super().__init__('Bollinger', labels)

    def add_feature(self, data):
        for i, period in enumerate(self.periods):
            label_top = self.label[2 * i]
            label_bot = self.label[2 * i + 1]
            SMA = pd.Series(data['close'].rolling(period, min_periods=period).mean())
            SD = pd.Series(data['close'].rolling(period, min_periods=period).std())
            BT = pd.Series(SMA + 2 * SD, name=label_top)
            BB = pd.Series(SMA - 2 * SD, name=label_bot)
            data = data.join(BT).join(BB)
        return data

class blank(Feature):
	def __init__(self, label=None):
		super().__init__('blank', label)

	def add_feature(self, data):
		ser = pd.Series([0.0] * len(data.index), name=self.label)
		data = data.join(ser)
		return data

class stochastic(Period_Feature):
    def __init__(self, periods=14, label=None):
        super().__init__('stochastic', periods, label)

    def add_feature(self, data):
        for i, period in enumerate(self.periods):
            label = self.label[i]
            min = data['low'].rolling(window=period).min()
            max = data['high'].rolling(window=period).max()
            close = data['close']
            SO = pd.Series((close - min) / (max - min) * 100, name=label)
            data = data.join(SO)
        return data

class pct_change(Period_Feature):
    def __init__(self, periods=1, label=None):
        super().__init__('PCT Change', periods, label)

    def add_feature(self, data):
        for i, period in enumerate(self.periods):
            label = self.label[i]
            pct = pd.Series(data['close'].pct_change(period), name=label)
            data = data.join(pct)
        return data

class ATR(Period_Feature):
    def __init__(self, periods=25, label=None):
        super().__init__('ATR', periods, label)

    def add_feature(self, data):
        for i, period in enumerate(self.periods):
            label = self.label[i]
            ATR1 = pd.Series(abs(data['high'] - data['low']), name = 'ATR1')
            ATR2 = pd.Series(abs(data['high'] - data['close'].shift()), name = 'ATR2')
            ATR3 = pd.Series(abs(data['low'] - data['close'].shift()), name = 'ATR3')
            ATR = pd.concat([ATR1, ATR2, ATR3], axis=1).max(axis=1)
            averages = pd.Series(ATR.rolling(window=period).mean(), name = label)
            data = data.join(averages)

        return data
