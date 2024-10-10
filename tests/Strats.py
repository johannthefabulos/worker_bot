''' Test all permutations given variables and a list of values


'''

import json
import random

import numpy as np


class Permutation:
    def __init__(self, *args, **kwargs):
        self.perms = 1              # Total number of possible permutations
        self.perms_list = []        # Number of options for each variable
        self.current_perm = 0       # Current permutation out of the total
        self.current_perm_list = [] # Current permutation for each variable
        self.all_params = []        # All possible values for each variable
        self.tracked_results = []   # All results for each value of each variable (dim = num variables x num possible values for each variable x num results at that value)
        self.kwarg_indices = {}     # Points the index of a variable to its kwarg name
        self.num_args = 0
        for arg in args:
            self._add_to_params(arg)

        for key, val in kwargs.items():
            self._add_to_params(val, key)

        self.results = [None] * self.perms

    def _add_to_params(self, arg, name=None):
        arg = self._convert_to_list(arg)
        length = len(arg)
        if name:
            self.kwarg_indices[len(self.perms_list)] = name
        else:
            self.num_args += 1
        self.perms_list.append(length)
        self.perms *= length
        self.all_params.append(arg)
        self.tracked_results.append([[] for i in range(length)])
        self.current_perm_list.append(0)

    def _convert_to_list(self, value):
        if type(value) is not list:
            try:
                value = list(value)
            except:
                value = [value]
        return value

    def get_params_from_perm(self, perm):
        if perm < 0:
            return None
        params = []
        kwarg_params = {}
        for i in range(len(self.all_params)):
            num = perm % self.perms_list[i]
            perm //= self.perms_list[i]
            params, kwarg_params = self._append_in_correct_list(params, kwarg_params, i, num)
            self.current_perm_list[i] = num
        params.append(kwarg_params)
        print(params)
        return params

    def next_perm(self):
        if self.current_perm >= self.perms:
            return None
        self.current_perm += 1
        return self.get_params_from_perm(self.current_perm - 1)

    def record_last_result(self, result):
        self.results[self.current_perm - 1] = result

        # Track each dimension
        for i in range(len(self.all_params)):
            num = self.current_perm_list[i]
            self.tracked_results[i][num].append(result)

    def is_done(self):
        return self.current_perm >= self.perms

    def _append_in_correct_list(self, params_list, kwarg_dict, idx, val):
        if idx in self.kwarg_indices:
            kwarg_dict[self.kwarg_indices[idx]] = self.all_params[idx][val]
        else:
            params_list.append(self.all_params[idx][val])
        return params_list, kwarg_dict

    def get_best_params(self):
        # Retrieve the best values from all results
        best_params = []
        kwarg_params = {}
        for i in range(len(self.tracked_results)):
            # Find the best result for this variable parameter
            param = self.tracked_results[i]
            best = sum(param[0])
            best_idx = 0
            for j in range(1, len(param)):
                results = param[j]
                current = sum(results)
                if current > best:
                    best = current
                    best_idx = j
            best_params, kwarg_params = self._append_in_correct_list(best_params, kwarg_params, i, best_idx)
        best_params.append(kwarg_params)
        return best_params

    def get_best_result(self):
        best = self.results[0]
        for result_idx in range(1, len(self.results)):
            result = self.results[result_idx]
            if result > best:
                best = result
        return best

    def __len__(self):
        return self.perms

class generic_strat:
    def __init__(self, name='generic strat'):
        self.name = name
        self.params = {}
        self.state = {'cur_idx':-1}
        self.state_hash = {}
        self.root_idx = 0

    def decision_funct(self, history):
        # Load state
        params_hash = hash(json.dumps(self.params, sort_keys=True))
        if self.state_hash:
            # Assign the current state
            self.state = self.state_hash.get(params_hash)
            if not self.state or self.state.get('cur_idx') != history.index[-1] - self.root_idx - 1:
                self.state = self.state_hash.get(-1).copy()
        else:
            # Assign the root state
            self.state_hash[-1] = self.state.copy()
            self.root_idx = history.index[-1]
        self.state['cur_idx'] = history.index[-1] - self.root_idx

        # Return the decision value
        decision = self.decision(history)

        # Save state
        self.state_hash[params_hash] = self.state
        return decision

    def decision(self, history):
        # Override this function to return a decision
        # -1 = SELL
        #  0 = HOLD
        #  1 = BUY
        return 1

    def __repr__(self):
        return self.name

class average_periods(generic_strat):
    def __init__(self, name='avg_periods'):
        super().__init__(name=name)
        self.params['avg_periods'] = [16, 32, 64, 120]

        self.max_periods = max(self.params['avg_periods']) # Maximum number of average periods
        self.state['last_buy'] = -1      # Number of periods since the last buy

    def compute_last_buy(self, history):
        # Given the history, returns the number of periods since there was a buy
        new_history = history.copy()
        first = True
        for i in range(self.max_periods):
            if not first:
                new_history.drop(new_history.tail(1).index,inplace=True)
            if self.buy_decision(new_history):
                # Returned true. Take count
                return i
            if first:
                first = False
        # Assume we would have bought at the beginning of all periods
        return len(history.index)


    def decision(self, history):
        # Run the actual decision function and add a holding period before selling
        periods_since_buy = self.state['last_buy']

        if periods_since_buy >= 0:
            decision = self.buy_decision(history)
            if decision:
                self.state['last_buy'] = 0
                return 1
            else:
                periods_since_buy += 1

        else:
            periods_since_buy = self.compute_last_buy(history)

        self.state['last_buy'] = periods_since_buy
        if periods_since_buy == 0:
            return 1
        if periods_since_buy < self.params['avg_periods']:
            return 0
        return -1

    def buy_decision(self, history, Limit=False):
        # Return true if we want to buy and false if we don't
        # This function is meant to be overridden
        return False

class increasing_in_a_row(average_periods):             
    def __init__(self, name='increasing_in_a_row'):
        super().__init__(name=name)

        up_down_4 = self.get_binary_permutations(4)
        up_down_5 = self.get_binary_permutations(5)
        up_down_6 = self.get_binary_permutations(6)
        up_down_7 = self.get_binary_permutations(7)
        up_down_8 = self.get_binary_permutations(8)
        self.params['up_down'] = up_down_7              
        self.params['bias'] = [0.0000, 0.0005, 0.001]

    def get_binary_permutations(self, n = 2, base = 2):
        perms = []
        for permutation in range(base ** n):
            this_perm = []
            for i in range(n):
                x = permutation % base          
                permutation //= base            
                this_perm = [x] + this_perm 
            perms.append(this_perm)
        return perms
    
    def buy_decision(self, history):
        up_down = self.params['up_down']
        max_decreasing = 0
        i = -len(up_down)
        for up in up_down:
            if up == 1:
                if history.iloc[i]['PCT Change_1'] + self.params['bias'] < 0:
                    return False
            else:
                if history.iloc[i]['PCT Change_1'] + self.params['bias'] > 0:
                    return False
            i += 1
        return True

class buy_if_red(average_periods):
    def __init__(self):
        super().__init__('buy_if_red')

    def buy_decision(self, history):
        if history.iloc[-1]['close'] > history.iloc[-2]['close']:      
                                                                        
            return True
        return False

class avg_strat(average_periods):
    def __init__(self):
        super().__init__('avg_strat')
        self.params['percent'] = [.96,.98,.99,.995]

    def buy_decision(self, history):
        percent = self.params['percent']
        moving_avg = history.iloc[-1]['SMA_19']
        current_price = history.iloc[-1]['close']
        percentage = moving_avg * percent
        if current_price <= percentage:
            return True
        return False

class rand_walk(average_periods):
    def __init__(self):
        super().__init__('rand_walk')
        self.params['buy_chance'] = [.1,.05,.01]

    def buy_decision(self, history):
        chance = self.params['buy_chance']
        if random.random() < chance:
            return True
        return False

class all_returns_signal(average_periods):
    def __init__(self):
        super().__init__('rand_walk')
    def buy_decision(self, history):
        return True

class large_gain_signal(average_periods):
    def __init__(self):
        super().__init__('large_gain_signal')

    def buy_decision(self, history):
        threshold = 1.05
        start = stats.describe(history.iloc[:-1]['close']).mean
        final = history.iloc[-1]['close']
        return final / start > threshold

class rsi_signal(average_periods):
    def __init__(self):
        super().__init__('rsi_signal')
        self.params ['threshold'] = [.1, .125, .15, .2]
    def buy_decision(self, history):
        threshold = self.params['threshold']
        return history.iloc[-1]['RSI_20'] < threshold

class bb_signal(generic_strat):
    def __init__(self):
        super().__init__('bb_signal')
        self.params['bias'] = [0.00, 0.0005, 0.001]
        self.params['negate'] = [True, False]

    def decision(self, history):
        bias = self.params['bias']
        negate = self.params['negate']
        final = history.iloc[-1]['close']
        bottom = history.iloc[-1]['Bollinger_B_20']
        top = history.iloc[-1]['Bollinger_T_20']
        if final > top * (1 + bias):
            return 1 if not negate else -1
        if final < bottom * (1 - bias):
            return -1 if not negate else 1
        return 0

class pct_change_signal(average_periods):
    def __init__(self):
        super().__init__('pct_change_signal')

    def buy_decision(self, history):
        return history.iloc[-2]['PCT Change_1'] - history.iloc[-1]['PCT Change_1'] < -0.02

class ema_signal(average_periods):
    def __init__(self):
        super().__init__('ema_signal')
        self.params['short_period'] = [5, 9]
        self.params['long_period'] = [19, 29]

    def buy_decision(self, history):
        short_term = history.iloc[-1]['EMA_9']
        long_term = history.iloc[-1]['EMA_19']
        return short_term < long_term

class lowest_price_signal(average_periods):
    def __init__(self):
        super().__init__('lowest_price_signal')
    def buy_decision(self, history):
        final = history.iloc[-1]['close']
        for i in range(len(history.index) - 1):
            if history.iloc[i]['close'] < final:
                return False
        return True

class highest_price_signal(average_periods):
    def __init__(self):
        super().__init__('highest_price_signal')

    def buy_decision(self, history):
        final = history.iloc[-1]['close']
        for i in range(len(history.index) - 1):
            if history.iloc[i]['close'] > final:
                return False
        return True

class best_price_signal(generic_strat):
    def __init__(self):
        super().__init__('best_price_signal')
        self.params['periods'] = [14]#[10, 14]
        self.params['min_count_up'] = [2]
        self.params['min_count_down'] = [2]
        self.params['only_sequential'] = [True]
        self.params['min_buy'] = [-130]
        self.params['min_sell'] = [120]
        self.params['stop_loss'] = [1]
        self.params['direction_periods'] = [4]
        self.state['up_in_a_row'] = -999
        self.state['down_in_a_row'] = -999
        self.state['sell_loss'] = -1
        self.state['can_buy'] = True
        self.state['adjusted'] = 0
        self.min_back = max(max(self.params['min_count_up']), max(self.params['min_count_down']))

    def setup_state(self, history):
        self.state['up_in_a_row'] = 0
        self.state['down_in_a_row'] = 0
        self.state['sell_loss'] = 0
        self.state['can_buy'] = True
        for i in range(-self.min_back, 0):
            self.decision(history.iloc[:i], print_stuff=False)

    def get_max_min(self, history):
        direction_periods = self.params['direction_periods']
        start = history.iloc[-direction_periods]['close']
        end = history.iloc[-1]['close']
        slope = (end - start) / direction_periods
        low = start
        high = start
        for i in range(1, direction_periods - 1):
            high += slope
            low += slope
            close = history.iloc[-direction_periods + i]['close']
            if close > high:
                high = close
            if close < low:
                low = close
        return low, high


    def decision(self, history, print_stuff=False):
        if self.state['sell_loss'] < 0 or self.state['up_in_a_row'] == -999 or self.state['down_in_a_row'] == -999:
            new_hist = history.copy()
            self.setup_state(new_hist)
        periods = self.params['periods']
        #direction_periods = self.params['direction_periods']
        #adjusted = self.state['adjusted']
        #if adjusted == 0:
        #    for i in range(1, direction_periods):
        #        low = min(history.iloc[-direction_periods - i:-i]['low'])
        #        high = max(history.iloc[-direction_periods - i:-i]['high'])
        #        last = history.iloc[-i - 1]['close']
        #        stochastic = (last - low) / (high - low) * 100
        #        adjusted += stochastic
        #else:
        #    adjusted *= (direction_periods - 1)
        #low = min(history.iloc[-periods:]['low'])
        #high = max(history.iloc[-periods:]['high'])
        last = history.iloc[-1]['close']
        low, high = self.get_max_min(history)
        if high != low:
            stochastic = (last - low) / (high - low) * 100
        else:
            stochastic = last / high * 100
        #adjusted += stochastic
        #adjusted /= direction_periods
        #self.state['adjusted'] = adjusted
        #print('Adjusted:', adjusted)
        #print('Stochastic:', stochastic)
        #print('close:', last)
        #stochastic -= (adjusted - 50)
        history.at[history.index[-1], 'blank'] = stochastic
        #print(stochastic)
        #print('Adjusted stochastic:', stochastic)
        #input()

        if stochastic >= self.params['min_sell']:
            self.state['down_in_a_row'] = 0
            self.state['up_in_a_row'] += 1
            if self.state['up_in_a_row'] >= self.params['min_count_up']:
                self.state['can_buy'] = True
                self.state['sell_loss'] = 0
                return -1
        elif not self.state['can_buy'] and last < self.state['sell_loss']:
            return -1
        elif self.state['can_buy'] and stochastic <= self.params['min_buy']:
            self.state['down_in_a_row'] += 1
            self.state['up_in_a_row'] = 0
            if self.state['down_in_a_row'] >= self.params['min_count_down']:
                self.state['can_buy'] = False
                self.state['sell_loss'] = last * (1 - self.params['stop_loss'])
                return 1
        else:
            self.state['down_in_a_row'] = 0
            self.state['up_in_a_row'] = 0
        return 0


        if min(history.iloc[-periods:-1]['close']) >= history.iloc[-1]['close']:
            self.state['down_in_a_row'] += 1
            self.state['up_in_a_row'] = 0
            if self.state['down_in_a_row'] >= self.params['min_count_down']:
                return 1
        elif max(history.iloc[-periods:-1]['close']) <= history.iloc[-1]['close']:
            self.state['down_in_a_row'] = 0
            self.state['up_in_a_row'] += 1
            if self.state['up_in_a_row'] >= self.params['min_count_up']:
                return -1
        elif not self.params['only_sequential']:
            self.state['up_in_a_row'] -= 1
            self.state['down_in_a_row'] -= 1
        else:
            self.state['up_in_a_row'] = 0
            self.state['down_in_a_row'] = 0
        if print_stuff:
            print('Last price:', history.iloc[-1]['close'])
            print('up:', self.state['up_in_a_row'])
            print('down:', self.state['down_in_a_row'])
        return 0

class lowest_rsi_signal(average_periods):
    def __init__(self):
        super().__init__('lowest_rsi_signal')

    def buy_decision(self, history):
        final = history.iloc[-1]['RSI_20']
        for i in range(len(history.index) - 1):
            if history.iloc[i]['RSI_20'] < final:
                return False
        return True

class lowest_n_price_signal(average_periods):
    def __init__(self):
        super().__init__('lowest_n_price_signal')
        self.params['n'] = [100]
        self.params['min_back'] = [10]
        self.params['n_in_a_row'] = [1, 2]
        self.params['bias'] = [0.0, -0.0002, -0.0005]

    def buy_decision(self, history):
        n = self.params['n']
        n_in_a_row = self.params['n_in_a_row']
        if history.index[-1] - np.argmin(history['close'].iloc[-n:-n_in_a_row]) < self.params['min_back']:
            return False
        if history.iloc[-n_in_a_row]['close'] < np.min(history['close'].iloc[-n:-n_in_a_row]):
            for i in range(n_in_a_row):
                if history.iloc[-i - 1]['PCT Change_1'] + self.params['bias'] > 0:
                    return False
            return True
        return False

class supertrend(generic_strat):
    def __init__(self):
        super().__init__('supertrend')
        self.params['multiplier_up'] = [2, 3, 4, 5, 6, 7, 8]
        self.params['multiplier_down'] = [2, 3, 4, 5, 6, 7, 8]
        self.state['last_final_upperband'] = -1
        self.state['last_final_lowerband'] = -1
        self.state['last_supertrend'] = 0
        self.state['last_side'] = 1
        self.min_back = 20

    def setup_state(self, history):
        self.state['last_final_upperband'] = 0
        self.state['last_final_lowerband'] = 0
        for i in range(-self.min_back, 0):
            self.decision(history.iloc[:i], print_stuff=False)

    def decision(self, history, print_stuff=False):

        if self.state['last_final_lowerband'] < 0 or self.state['last_final_upperband'] < 0:
            new_hist = history.copy()
            self.setup_state(new_hist)

        high = history.iloc[-1]['high']
        low = history.iloc[-1]['low']
        atr = history.iloc[-1]['ATR_25']
        #print('ATR:', atr)
        close = history.iloc[-1]['close']
        basic_upperband = (high + low) / 2 + self.params['multiplier_up'] * atr
        basic_lowerband = (high + low) / 2 - self.params['multiplier_down'] * atr
        #print('Upper:', basic_upperband)
        #print('Lower:', basic_lowerband)

        if self.state['last_final_upperband'] == 0 or (basic_upperband < self.state['last_final_upperband'] or history.iloc[-2]['close'] > self.state['last_final_upperband']):
            final_upperband = basic_upperband
        else:
            final_upperband = self.state['last_final_upperband']

        if self.state['last_final_lowerband'] == 0 or (basic_lowerband > self.state['last_final_lowerband'] or history.iloc[-2]['close'] < self.state['last_final_lowerband']):
            final_lowerband = basic_lowerband
        else:
            final_lowerband = self.state['last_final_lowerband']
        self.state['last_final_upperband'] = final_upperband
        self.state['last_final_lowerband'] = final_lowerband

        if self.state['last_side'] == 1 and close < final_upperband:
            supertrend = final_upperband
            #print('Up super:', supertrend)
            history.at[history.index[-1], 'blank'] = supertrend
            self.state['last_supertrend'] = supertrend
            return -1
        elif self.state['last_side'] == 1 and close > final_upperband:
            supertrend = final_lowerband
            #print('Down super:', supertrend)
            self.state['last_side'] = -1
            history.at[history.index[-1], 'blank'] = supertrend
            self.state['last_supertrend'] = supertrend
            return 1
        elif self.state['last_side'] == -1 and close < final_lowerband:
            supertrend = final_upperband
            #print('Up super:', supertrend)
            self.state['last_side'] = 1
            history.at[history.index[-1], 'blank'] = supertrend
            self.state['last_supertrend'] = supertrend
            return -1
        elif self.state['last_side'] == -1 and close > final_lowerband:
            supertrend = final_lowerband
            #print('Down super:', supertrend)
            history.at[history.index[-1], 'blank'] = supertrend
            self.state['last_supertrend'] = supertrend
            return 1
        else:
            history.at[history.index[-1], 'blank'] = self.state['last_supertrend']
            return 0

class lowest_n_rsi_signal(average_periods):
    def __init__(self):
        super().__init__('lowest_n_rsi_signal')
        self.params['n'] = [2, 3]

    def buy_decision(self, history):
        n = self.params['n']
        last_n = list(history.iloc[-n:]['RSI_20'])
        sorted_last = last_n.copy()
        sorted_last.sort(reverse=True)
        if last_n != sorted_last:
            return False

        for i in range(len(history.index) - n):
            for value in last_n:
                if history.iloc[i]['RSI_20'] < value:
                    return False
        return True

class stochastic_strat(generic_strat):
    def __init__(self):
        super().__init__('stochastic_strat')
        self.params['limit_up'] = [90, 95, 100]
        self.params['limit_down'] = [0, 5, 15]
        self.params['avg_periods'] = [1]
        self.params['await_turnaround'] = [False, True]
        self.params['trend_periods'] = [20]
        self.params['slope_coeff'] = [-5000]
        self.state['set_up'] = False
        self.state['set_down'] = False
        self.min_back = 80
        self.super = supertrend()
        self.super.params['multiplier_up'] = 10
        self.super.params['multiplier_down'] = 10

    def decision(self, history, print_stuff=False):
        periods = self.params['avg_periods']
        wait = self.params['await_turnaround']
        trend_periods = self.params['trend_periods']
        #trend = np.mean([history.iloc[-periods - i - 1:-i - 1]['stochastic_14'].mean() for i in range(self.params['trend_periods'])])
        trend = ((history.iloc[-1]['close'] - history.iloc[-trend_periods - 1]['close']) / history.iloc[-trend_periods - 1]['close'])
        #slope = (history.iloc[-1]['close'] / history.iloc[-periods - 1]['close'] - 1) / periods
        previous = history.iloc[-periods - 1:-1]['stochastic_14'].mean()
        current = history.iloc[-periods:]['stochastic_14'].mean()
        offset = trend * self.params['slope_coeff']
        previous += offset
        current += offset
        #print(offset)

        can_buy = True#(self.super.decision(history) == 1)
        can_sell = True#(self.super.decision(history) == -1)
        history.at[history.index[-1], 'blank'] = current
        if not can_buy:
            self.state['set_down'] = False
        if not can_sell:
            self.state['set_up'] = False
        #print('up:', self.state['set_up'])
        #print('down:', self.state['set_down'])
        #print('prev:', previous)
        #print('cur:', current)
        if wait:
            if can_sell and self.state['set_up'] and current < previous:
                self.state['set_up'] = False
                return -1
            elif can_buy and self.state['set_down'] and current > previous:
                self.state['set_down'] = False
                return 1
            elif can_sell and not self.state['set_up'] and current > self.params['limit_up']:
                self.state['set_up'] = True
            elif can_buy and not self.state['set_up'] and current < self.params['limit_down']:
                self.state['set_down'] = True
            return 0
        else:
            if can_sell and current > self.params['limit_up']:
                return -1
            elif can_buy and current < self.params['limit_down']:
                return 1
            return 0


class up_down_stat(average_periods):
    def __init__(self, name='up_down_stat'):
        super().__init__(name=name)
        self.params['train_periods'] = [15000]
        self.params['avg_periods'] = [128]
        self.params['min_buy'] = [1]
        self.params['buy_sell_threshold'] = [0.04, 0.05, 0.06]
        self.params['n'] = [5]
        self.state['initial'] = True

    def init_perms(self, history, n = 2, base = 3):
        self.state['data'] = {'buy':0,'up':0,'expected':1}
        perms = []
        for permutation in range(base ** n):
            this_perm = []
            for i in range(n):
                x = permutation % base
                permutation //= base
                this_perm = [x] + this_perm
            perm = tuple(this_perm)
            self.state['data'][perm] = {'buy':0,'up':0,'expected':1}

        for i in range(self.params['avg_periods'] + self.params['n'] + 1, len(history.index)):
            self.update_data(history.iloc[:i])

    def get_current_perm(self, history):
        up_down = []
        for i in range(-self.params['n'], 0):
            if abs(history.iloc[i]['PCT Change_1']) < abs(history.iloc[i - 1]['PCT Change_1']):
                up_down.append(1)
            elif history.iloc[i]['PCT Change_1'] > 0:
                up_down.append(2)
            else:
                up_down.append(0)
        return tuple(up_down)

    def add_remove_decision(self, history, add):
        added = 1 if add else -1
        perm = self.get_current_perm(history.iloc[-self.params['avg_periods'] - self.params['n'] - 1:-self.params['avg_periods']])
        data_total = self.state['data']['buy']
        perm_total = self.state['data'][perm]['buy']
        self.state['data'][perm]['buy'] += added
        self.state['data']['buy'] += added
        expected = history.iloc[-1]['close'] / history.iloc[-self.params['avg_periods']-1]['close'] - 0.00075
        if add:
            self.state['data']['expected'] = (self.state['data']['expected'] * data_total + expected) / (data_total + 1)
            self.state['data'][perm]['expected'] = (self.state['data'][perm]['expected'] * perm_total + expected) / (perm_total + 1)
        else:
            if data_total > 1:
                self.state['data']['expected'] = (self.state['data']['expected'] * data_total - expected) / (data_total - 1)
            if perm_total > 1:
                self.state['data'][perm]['expected'] = (self.state['data'][perm]['expected'] * perm_total - expected) / (perm_total - 1)
        if expected > 1:
            self.state['data'][perm]['up'] += added
            self.state['data']['up'] += added


    def update_data(self, history):
        # First, remove the data from train_periods back through train_periods + avg_periods (if we have enough periods)
        if len(history.index) > self.params['train_periods'] + self.params['avg_periods'] + self.params['n'] + 1:
            self.add_remove_decision(history.iloc[:-self.params['train_periods']], False)

        # Next, add the data from avg_periods back
        self.add_remove_decision(history, True)

    def buy_decision(self, history):
        print_stuff = False
        new_hist = history.copy()

        if self.state['initial']:
            self.init_perms(new_hist, n=self.params['n'])
            self.state['initial'] = False

        else:
            self.update_data(new_hist)

        perm = self.get_current_perm(new_hist)
        info = self.state['data'][perm]
        if info['buy'] > 0:
            proportion = info['up'] / info['buy']
        else:
            proportion = 0.0
        history.at[history.index[-1], 'blank'] = proportion
        if self.state['data']['buy'] < self.params['train_periods']:
            return 0
        expected = self.state['data']['expected']

        if print_stuff:
            print('Price:', history.iloc[-1]['close'])
            print('Buy:', self.state['data']['buy'])
            print('Up:', self.state['data']['up'])
            print('expected:', expected)
            print('Perm:', perm)
            print('Info:', info)
        if info['buy'] > self.params['min_buy']:
            total_prop = self.state['data']['up'] / self.state['data']['buy']
            perm_expected = info['expected']
            history.at[history.index[-1], 'blank'] = perm_expected - 1.0
            if print_stuff:
                print('Total proportion:', total_prop)
                print('Proportion:', proportion)
                print('Perm expected:', perm_expected)
            if perm_expected - expected > self.params['buy_sell_threshold']:
                if print_stuff:
                    print('Buy')
                return 1
            #elif total_prop - proportion > self.params['buy_sell_threshold']:
            #    if print_stuff:
        #            print('Sell')
    #                input()
#                return -1
        if print_stuff:
            print('None')
        return 0
