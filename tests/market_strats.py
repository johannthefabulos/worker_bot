import os
import sys

sys.path.insert(0,os.getcwd())
import ast
import copy
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from functools import partial

import numpy as np
import pandas as pd
import pandas_ta as ta

from exchanges import Exchange
from exchanges.Exchange import Candle
from manage.managetrades import get_amount, suggested_trade

#from manage.managetrades import stored_trade
from tests.Strats import Permutation, generic_strat

# import slices




class support():
    def __init__(self, num_periods_passed, support_price):
        self.periods_passed = num_periods_passed
        self.support_price = support_price
        self.sell_price = None
        self.buy_price = None
        self.bought = False
        self.sold = False
        self.periods_since_buy = None

class generic_market_strat(generic_strat):
    def __init__(self, name):
        super().__init__(name)
        self.params['market'] = True
        self.universal_custom_id = 0

    def initialize_amt_objs(self, number_amt_objs, symbol, manager_id):
        return_list_of_amt_objs = []
        for i in range(number_amt_objs):
            # This used to be the final argument of get_amount
            # additional_info=['tiers', str(i)]
            # Think I'll have to make it so that it has like a self.params thing
            current_obj = get_amount(symbol=symbol, amount_to_manage=None, custom_id=self.universal_custom_id, manager_id=manager_id)
            self.universal_custom_id+=1
            return_list_of_amt_objs.append(current_obj)
        return return_list_of_amt_objs

    def get_list_of_suggested_trades(self, list_of_objs):
        all_lists_of_suggested_trades = []
        total_amt = 0
        for amt_obj in list_of_objs:
            amt_object_suggested_trades = amt_obj.get_size_buy_sell()
            for trade in amt_object_suggested_trades:
                total_amt += Decimal(trade.total_amt)
            all_lists_of_suggested_trades.append(amt_object_suggested_trades)
        return all_lists_of_suggested_trades, total_amt


    def decision(self, history):
        #Overide decision to determine buy and sell prices
        return 0, 999999999

class similarity_test(generic_market_strat):
    def __init__(self, exchange, account, manager_id, symbol, percent_to_manage):
        super().__init__('similarity_test_strat')
        self.params['lookback'] = 5
        # Activate is seconds into the next minute you want to wait until processing decision
        self.params['activate'] = 1
        self.params['lag_periods'] = 2
        self.params['training'] = True
        self.params['prediction_periods'] = 8
        self.list_of_accepted_currencies = ['USDT', 'BTC', 'USDC']
        self.exchange = exchange
        self.account = account
        self.manager_id = manager_id
        self.symbol = symbol
        self.percent_to_manage = percent_to_manage
        
    def predict(self, last_periods):
        # hmm = last_periods
        # whaaa = self.percent_up
        # final = self.percent_up[last_periods]
        # what = tuple(last_periods[-self.params['prediction_periods'])
        return self.percent_up[tuple(last_periods[-self.params['prediction_periods']:])]

    def run_once(self):
        amount, _, _= self.account.amount_of_total_funds_to_handle(percent=self.percent_to_manage, end_currency='BTC', list_of_accepted_currencies=self.list_of_accepted_currencies, current_price=self.exchange.get_current_price(self.symbol), manager_id = self.manager_id)
        self.data = pd.read_parquet('all_data/BTCUSDT')
        self.data.index = pd.to_datetime(self.data['time'], unit='ms')
        self.candles = self.data['price'].resample(str(self.params['lookback'])+'min').ohlc()
        self.candles['return'] = self.candles['close'] / self.candles['close'].shift(self.params['lag_periods'])
        self.candles = self.candles.dropna()
        self.candles['direction'], self.bins = pd.qcut(self.candles['return'], q=2, labels=[-1, 1], retbins=True)
        self.frequencies = defaultdict(int)
        self.percent_up = defaultdict(float)
        # try:
        df = pd.read_csv('ugh.csv')
        df['Key'] = df['Key'].apply(lambda x: ast.literal_eval(x))
        df['Value'] = df['Value'].astype(int)
        self.frequencies = dict(zip(df['Key'], df['Value']))

        df2 = pd.read_csv('ugh1.csv')
        df2['Key'] = df2['Key'].apply(lambda x: ast.literal_eval(x))
        df2['Value'] = df2['Value'].astype(float)
        self.percent_up = dict(zip(df2['Key'], df2['Value']))
        # except:
        #     for i in range(len(self.candles.index)):
        #         if i - self.params['prediction_periods'] < 0:
        #             continue
        #         key = tuple(self.candles.iloc[i-self.params['prediction_periods']:i]['direction'])
        #         direction = self.candles.iloc[i]['direction']
        #         self.percent_up[key] = (self.frequencies[key] * self.percent_up[key] + direction) / (self.frequencies[key] + 1)
        #         self.frequencies[key] += 1
        #     df = pd.DataFrame(list(self.frequencies.items()), columns=['Key', 'Value'])
        #     df2 = pd.DataFrame(list(self.percent_up.items()), columns=['Key', 'Value'])
        #     df.to_csv('ugh.csv')
        #     df2.to_csv('ugh1.csv')
            # my_pd = pd.DataFrame.from_records(self.frequencies)
            # my_pd.to_csv('ugh.csv')
        self.list_of_objs = self.initialize_amt_objs(number_amt_objs=1, symbol=self.symbol, manager_id=self.manager_id)
        for idx, amt_obj in enumerate(self.list_of_objs):
            amt_obj.append_additional_info(key='similarity', element = idx)
            amt_obj.set_amount_to_manage_setter(amount)
    
    def get_direction(self, last_x_candles):
        last_x_candles['direction'] = pd.cut(last_x_candles['return'], bins=self.bins, labels=[-1, 1])
        return list(last_x_candles['direction'])

    def get_all_suggested_trades_one_list(self):
        all_suggested_trades, total_amt = self.get_list_of_suggested_trades(self.list_of_objs)
        suggested_trades = []
        for every_list in all_suggested_trades:
            for trade in every_list:
                suggested_trades.append(trade)
        return suggested_trades, suggested_trades

    def decision(self):
        amount, _, _= self.account.amount_of_total_funds_to_handle(percent=self.percent_to_manage, end_currency='BTC', list_of_accepted_currencies=self.list_of_accepted_currencies, current_price=self.exchange.get_current_price(self.symbol), manager_id = self.manager_id)
        for idx, amt_obj in enumerate(self.list_of_objs):
            amt_obj.set_amount_to_manage_setter(amount)
        for obj in self.list_of_objs:
            _, orders_editable = self.get_all_suggested_trades_one_list()
            prediction_candles = self.exchange.get_last_candles(self.symbol, self.params['prediction_periods'] + self.params['lag_periods'], self.params['lookback'])
            candles_pd = pd.DataFrame.from_dict(candle.to_dict() for candle in prediction_candles)
            candles_pd['return'] = candles_pd['close'] / candles_pd['close'].shift(self.params['lag_periods'])
            candles_pd = candles_pd[self.params['lag_periods']:]
            last_directions = self.get_direction(candles_pd)
            prediction = self.predict(last_directions)
            for order in orders_editable:
                if order.side == 'buy':
                    if prediction > 0:
                        self.exchange.market_order(order)
                if order.side == 'sell':
                    if prediction < 0:
                        self.exchange.market_order(order)
                
    def order_buy_handler(self, order):
        pass
    
    def order_sell_handler(self, order):
        pass

    

class market_strat_sentiment_slices(generic_market_strat):
    def __init__(self, exchange, account, manager_id, symbol, percent_to_manage):

        super().__init__('market_strat_support')
        self.params['lookback'] = 5
        self.params['furthest_back'] = [288]
        self.params['strong_previous'] = [287]
        self.params['medium_previous'] = [200]
        self.params['weak_previous'] = [50]
        self.params['percent_strong'] =[.5, .6, .7, .8]
        # self.params['percent_strong'] =[.6]
        # self.params['medium_left'] =[.66, .77, .88, .95]
        self.params['medium_left'] =[.3]
        self.params['timeout_periods'] = [5]
        # self.params['percent_above_window'] = [.1, .05, .15]
        self.params['percent_above_window'] = [.1]
        self.manager_id = manager_id
        self.params['num_periods_after'] = [5]
        self.list_of_accepted_currencies = ['USD', 'USDT', 'BTC', 'ETH']
        self.percent_to_manage = percent_to_manage
        self.exchange = exchange
        self.account = account
        self.manager_id = manager_id
        self.symbol = symbol
        self.list_of_objs = []
        self.support_system = {'strong':support(float('inf'), float('inf')), 'medium':support(float('inf'), float('inf')), 'weak':support(float('inf'), float('inf')) }
        self.final_volume = None
        self.high_window = 0

    def run_once(self):
        amount, _, _= self.account.amount_of_total_funds_to_handle(percent=self.percent_to_manage, end_currency='BTC', list_of_accepted_currencies=self.list_of_accepted_currencies, current_price=self.exchange.get_current_price(self.symbol), manager_id = self.manager_id)
        self.amt_to_manage = amount
        # TODO make this function run first in sim_test or what was sim_test
        #Like arduino there is a section for things to run only once to initialize things, I think that is important here as well
        # self.final_price_and_volume = self.get_info_for_tiers(now_iso_format=self.now_iso_format, amt_to_manage=self.amt_to_manage)
        self.final_volume= self.get_info_for_tiers(self.amt_to_manage)
        self.list_of_objs = self.initialize_amt_objs(number_amt_objs=3, symbol=self.symbol, manager_id=self.manager_id)
        all_keys = list(self.support_system.keys())
        for idx, amt_obj in enumerate(self.list_of_objs):
            amt_obj.append_additional_info(key='support', element = all_keys[idx])
            amt_obj.set_amount_to_manage_setter(self.final_volume[all_keys[idx]])

    def decision(self):
        client, newsclient = slices.setup_apis()
    def order_buy_handler(self, order):
        pass
    def order_sell_handler(self):
        pass

class market_strat_support(generic_market_strat):
    def __init__(self, exchange, account, manager_id, symbol, percent_to_manage):
        super().__init__('market_strat_support')
        self.params['lookback'] = 5
        self.params['furthest_back'] = [288]
        self.params['strong_previous'] = [287]
        self.params['medium_previous'] = [200]
        self.params['weak_previous'] = [50]
        self.params['percent_strong'] =[.5, .6, .7, .8]
        # self.params['percent_strong'] =[.6]
        # self.params['medium_left'] =[.66, .77, .88, .95]
        self.params['medium_left'] =[.3]
        self.params['timeout_periods'] = [5]
        # self.params['percent_above_window'] = [.1, .05, .15]
        self.params['percent_above_window'] = [.1]
        self.manager_id = manager_id
        self.params['num_periods_after'] = [5]
        self.list_of_accepted_currencies = ['USD', 'USDT', 'BTC', 'ETH']
        self.percent_to_manage = percent_to_manage
        self.exchange = exchange
        self.account = account
        self.manager_id = manager_id
        self.symbol = symbol
        self.list_of_objs = []
        self.support_system = {'strong':support(float('inf'), float('inf')), 'medium':support(float('inf'), float('inf')), 'weak':support(float('inf'), float('inf')) }
        self.final_volume = None
        self.high_window = 0

    def get_info_for_tiers(self, amt_to_manage):
        all_keys = list(self.support_system.keys())
        for key in all_keys:
            self.support_system[key].periods_passed = float('inf')
            self.support_system[key].support_price = float('inf')
        self.amt_to_manage = amt_to_manage
        total_candles = self.exchange.get_last_candles(self.symbol, self.params['strong_previous'], self.params['lookback'])
        length_candles = len(total_candles)
        for idx, candle in enumerate(total_candles):
            if candle.high > self.high_window:
                self.high_window = candle.high
            if candle.low < self.support_system['strong'].support_price:
                self.support_system['strong'].support_price = candle.low
                self.support_system['strong'].periods_passed = (length_candles - idx)
            if idx >= (length_candles - self.params['medium_previous']) and candle.low < self.support_system['medium'].support_price:
                self.support_system['medium'].support_price = candle.low 
                self.support_system['medium'].periods_passed = (length_candles - idx)
            if idx >= (length_candles - self.params['weak_previous']) and candle.low < self.support_system['weak'].support_price:
                self.support_system['weak'].support_price = candle.low
                self.support_system['weak'].periods_passed = (length_candles - idx)
        strong_amt = self.amt_to_manage * Decimal(self.params['percent_strong'])
        medium_amt = strong_amt * Decimal(self.params['medium_left'])
        weak_amt = self.amt_to_manage - Decimal(strong_amt + medium_amt)
        if weak_amt < 0:
            raise ValueError('this should never be negative')
        final_volume = {'strong':strong_amt,'medium':medium_amt,'weak':weak_amt}
        return final_volume

    def get_prices(self):
        all_keys = list(self.support_system.keys())
        for key in all_keys:
            self.support_system[key].buy_price = self.support_system[key].support_price  + (self.params['percent_above_window'] * (self.high_window - self.support_system[key].support_price))
            if self.support_system[key].bought and self.support_system[key].sell_price is not None:
                if self.support_system[key].sell_price < self.support_system[key].support_price:
                    self.support_system[key].sell_price = self.support_system[key].support_price
            else:
                self.support_system[key].sell_price = self.support_system[key].support_price

    def get_all_suggested_trades_one_list(self):
        all_suggested_trades, total_amt = self.get_list_of_suggested_trades(self.list_of_objs)
        suggested_trades = []
        for every_list in all_suggested_trades:
            for trade in every_list:
                suggested_trades.append(trade)
        return suggested_trades, suggested_trades

    def run_once(self):
        amount, _, _= self.account.amount_of_total_funds_to_handle(percent=self.percent_to_manage, end_currency='BTC', list_of_accepted_currencies=self.list_of_accepted_currencies, current_price=self.exchange.get_current_price(self.symbol), manager_id = self.manager_id)
        self.amt_to_manage = amount
        # TODO make this function run first in sim_test or what was sim_test
        #Like arduino there is a section for things to run only once to initialize things, I think that is important here as well
        # self.final_price_and_volume = self.get_info_for_tiers(now_iso_format=self.now_iso_format, amt_to_manage=self.amt_to_manage)
        self.final_volume= self.get_info_for_tiers(self.amt_to_manage)
        self.list_of_objs = self.initialize_amt_objs(number_amt_objs=3, symbol=self.symbol, manager_id=self.manager_id)
        all_keys = list(self.support_system.keys())
        for idx, amt_obj in enumerate(self.list_of_objs):
            amt_obj.append_additional_info(key='support', element = all_keys[idx])
            amt_obj.set_amount_to_manage_setter(self.final_volume[all_keys[idx]])

    def decision(self):
        self.get_info_for_tiers(self.amt_to_manage)
        _, orders_editable = self.get_all_suggested_trades_one_list()
        for order in orders_editable:
            if order.side == 'sell':
                self.support_system[order.additional_info['support']].periods_since_buy = self.exchange.periods_passed_now_timestamp(order.parent_timestamp, self.params['lookback'])
            else:
                self.support_system[order.additional_info['support']].periods_since_buy = None
            self.get_prices()
            print(order.side)
            current_price = self.exchange.get_current_price(self.symbol)
            current_price = Decimal(current_price)
            print(current_price <= self.support_system[order.additional_info['support']].buy_price)
            if order.side == 'buy' and current_price <= self.support_system[order.additional_info['support']].buy_price and self.support_system[order.additional_info['support']].periods_passed >= 10 and self.support_system[order.additional_info['support']].periods_passed <= 15 and order.placed != True:
                self.exchange.market_order(order)
                self.support_system[order.additional_info['support']].bought = True
            if order.side == 'sell' and current_price >= self.support_system[order.additional_info['support']].sell_price and self.support_system[order.additional_info['support']].periods_since_buy >= self.params['timeout_periods'] and order.placed != True:
                self.exchange.market_order(order)
                self.support_system[order.additional_info['support']].bought = False

    def order_buy_handler(self, order):
        self.decision()
    
    def order_sell_handler(self, order):
        self.decision()
           