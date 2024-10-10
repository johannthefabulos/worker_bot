import os
import sys
from threading import Lock

sys.path.insert(0,os.getcwd())
import copy
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from functools import partial

import numpy as np
import pandas as pd

from exchanges import Exchange
from exchanges.Exchange import Candle
from manage.managetrades import get_amount, suggested_trade

#from manage.managetrades import stored_trade
from tests.Strats import Permutation, generic_strat


class generic_limit_strat(generic_strat):
    def __init__(self, name):
        super().__init__(name)
        self.params['limit'] = True
        self.universal_custom_id = 0

    # def create_amt_objs(self, ):
    #     list_of_results = []
    #     for single_function in list_of_functions:
    #         list_of_results.append(single_function())
    #     return list_of_results
    #     # self.buy_tiers, self.sell_tiers = self.get_tiers(self.params['buy_tiers'], self.params['max_buydown'], fucn)
    #     # self.list_of_amt_objs = self.preliminary_create_empty_amt_objs(tiers_length=self.params['buy_tiers'], symbol=self.symbol, custom_id=self.custom_id, manager_id=self.set_manager_id)

    def sell_decay(self, first_price, decay_time, rate, period_length):
        print('second sell')
        total_periods_passed = self.exchange.periods_passed_now_timestamp(decay_time, period_length)
        new_price = first_price * Decimal(pow(rate, int(total_periods_passed)))
        return new_price

    def helper_total_periods_passed(self, date_time_object_start, date_time_object_end, period_len_minutes):
        time_passed_delta = date_time_object_end - date_time_object_start
        return ((time_passed_delta.total_seconds()/60)//period_len_minutes)

    def get_amount_currency_available(self, account, percent, time, end_currency, list_of_accepted):
        amount, current_price, total = account.amount_of_total_funds_to_handle(percent=percent, time_iso=time, end_currency=end_currency, list_of_accepted_currencies=list_of_accepted)
        return amount, current_price, total

    def verify_amount_allowed(self, amt_allowed, list_of_amt_objs):
        total_addition = 0
        for amt_obj in list_of_amt_objs:
            _, obj_amt_to_manage = amt_obj.consider_amt()
            total_addition += obj_amt_to_manage
        if abs(total_addition-amt_allowed) >= self.TOLERANCE:
            raise ValueError('Too much or too little allocated')
    
    def get_ema(self, symbol, num_prev_candles, len_period_minutes):
        #TODO:make self.exchange.get_last_candles work for both real and fake exchange
        #TODO: make sure this actually calculates correctly
        candles = self.exchange.get_last_candles(symbol, num_prev_candles, len_period_minutes)        
        candles_df = pd.DataFrame.from_records(candle.to_dict() for candle in candles)
        while len(candles_df.index) < num_prev_candles + 1:
            #TODO: pretty sure this is just adding the random last line element to add length...not sure why its needed but I'll leave it here for now
            last_line = candles_df.iloc[-1]
            candles_df = candles_df.append(last_line)
        # The actual ema calculation
        averages = pd.Series(candles_df['open'].ewm(span=num_prev_candles, min_periods=num_prev_candles).mean(), name='EMA')
        # Gets only the last average which is the most recent
        final_average = averages.iloc[-1]
        return final_average

    def sort_through_trades(self, one_list, amt_obj):
        amt_obj_trades = []
        the_ids = []
        for idx, trade in enumerate(one_list):
            if trade.amt_id == amt_obj.amt_obj_id:
                the_ids.append(idx)
                amt_obj_trades.append(trade)
        return amt_obj_trades, the_ids

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
    
class advantage(generic_limit_strat):
    def __init__(self, exchange, account, manager_id, symbol, percent_to_manage):
        super().__init__('avg_limit_strat')
        self.exchange = exchange
        self.account = account
        self.manager_id = manager_id
        self.symbol = symbol
        self.percent_to_manage = percent_to_manage
        self.percent_to_manage = .5
        self.params['lookback'] = 60
        self.params['activate'] = 1
        self.params['tiers'] = 3
        self.params['percentage_high'] = .6
        self.params['percent_below'] = .01
        self.params['percent_increase'] = [1]
        self.labels =[]

        self.params['']
        self.params['']
        self.list_of_accepted_currencies = ['USD', 'USDT', 'BTC', 'ETH', 'USDC']


    def calculate_percentages(self, amt):
        amts_list = []
        amts_list.append()
        left = 1-self.params['percentage_high']
        high_amt = amt * self.params['percentage_high']
        amts_list.append(high_amt)
        num_left = self.params['tiers'] - 1
        divided = Decimal(left/num_left)
        for i in range(num_left):
            amts_list.append(i)
        assert len(amts_list) == self.params['tiers']
        for idx, element in enumerate(amts_list):
            if idx == len(amts_list):
                self.labels.append('most_tier')
            self.labels.append('equal_tier:' + idx)
        return amts_list

    def run_once(self):
        amount, _, _= self.account.amount_of_total_funds_to_handle(percent=self.percent_to_manage, end_currency='USDT', list_of_accepted_currencies=self.list_of_accepted_currencies, current_price=self.exchange.get_current_price(self.symbol), manager_id = self.manager_id)
        self.amt_to_manage = amount
        # TODO make this function run first in sim_test or what was sim_test
        #Like arduino there is a section for things to run only once to initialize things, I think that is important here as well
        # self.final_price_and_volume = self.get_info_for_tiers(now_iso_format=self.now_iso_format, amt_to_manage=self.amt_to_manage)
        self.list_of_objs = self.initialize_amt_objs(number_amt_objs=self.params['tiers'], symbol=self.symbol, manager_id=self.manager_id)
        all_amts = self.calculate_percentages(amount)
        for idx, amt_obj in enumerate(self.list_of_objs):
            amt_obj.append_additional_info(key=self.labels[idx], element = idx)
            amt_obj.set_amount_to_manage_setter(all_amts[idx])

    def decision(self):
        _, orders_editable = self.get_all_suggested_trades_one_list()
        current_price = self.exchange.get_current_price(self.symbol)
        last_candle = self.exchange.get_last_candles(self, self.symbol, 1, self.params['lookback'])
        last_high = last_candle.high
        for order in orders_editable:
            multiplier = (order.amt_id + 1)
            if order.side == 'sell':
                if not order.active:
                    sell_price = last_high
                    order.original_price = sell_price
                    order.price = sell_price
                    self.exchange.new_order(order)
                else:
                    sell_price = last_high
                    order.price = sell_price
                    self.exchange.replace_order(order)
            if order.side == 'buy':
                if not order.active:
                    buy_price = last_high-(last_high*(self.params['percent_below']*multiplier))
                    order.original_price = buy_price
                    order.price = buy_price
                    self.exchange.new_order(order)
                else:
                    buy_price = last_high-(last_high*self.params['percent_below'])
                    order.price = buy_price
                    self.exchange.replace_order(order)

    def order_sell_handler(self, order):
        current_price = self.exchange.get_current_price(self.symbol)
        last_candle = self.exchange.get_last_candles(self, self.symbol, 1, self.params['lookback'])
        last_high = last_candle.high
        if not order.active:
            sell_price = last_high
            order.original_price = sell_price
            order.price = sell_price
            self.exchange.new_order(order)

    def order_buy_handler(self, order):
        multiplier = (order.amt_id + 1)
        current_price = self.exchange.get_current_price(self.symbol)
        last_candle = self.exchange.get_last_candles(self, self.symbol, 1, self.params['lookback'])
        last_high = last_candle.high
        buy_price = last_high-(last_high*(multiplier *self.params['percent_below']))
        if not order.active:
            order.price = buy_price
            self.exchange.new_order(order)

class test_strat(generic_limit_strat):
    def __init__(self, exchange, account, manager_id, symbol, percent_to_manage):
        super().__init__('avg_limit_strat')
        self.exchange = exchange
        self.account = account
        self.manager_id = manager_id
        self.symbol = symbol
        self.percent_to_manage = percent_to_manage
        self.percent_to_manage = .5
        self.params['lookback'] = 5
        self.params['activate'] = 1
        self.list_of_accepted_currencies = ['USD', 'USDT', 'BTC', 'ETH', 'USDC']

    def run_once(self):
        amount, _, _= self.account.amount_of_total_funds_to_handle(percent=self.percent_to_manage, end_currency='BTC', list_of_accepted_currencies=self.list_of_accepted_currencies, current_price=self.exchange.get_current_price(self.symbol), manager_id = self.manager_id)
        
        self.amt_to_manage = amount
        # TODO make this function run first in sim_test or what was sim_test
        #Like arduino there is a section for things to run only once to initialize things, I think that is important here as well
        # self.final_price_and_volume = self.get_info_for_tiers(now_iso_format=self.now_iso_format, amt_to_manage=self.amt_to_manage)
        self.list_of_objs = self.initialize_amt_objs(number_amt_objs=1, symbol=self.symbol, manager_id=self.manager_id)
        for idx, amt_obj in enumerate(self.list_of_objs):
            amt_obj.append_additional_info(key='test', element= idx)
            amt_obj.set_amount_to_manage_setter(amount)

    def get_all_suggested_trades_one_list(self):
        all_suggested_trades, total_amt = self.get_list_of_suggested_trades(self.list_of_objs)
        suggested_trades = []
        for every_list in all_suggested_trades:
            for trade in every_list:
                suggested_trades.append(trade)
        return suggested_trades, suggested_trades

    def decision(self):
        _, orders_editable = self.get_all_suggested_trades_one_list()
        current_price = self.exchange.get_current_price(self.symbol)
        current_price = Decimal(current_price)
        for order in orders_editable:
            if order.side == 'sell':
                if not order.active:
                    sell_price = current_price + 50
                    order.original_price = sell_price
                    order.price = sell_price
                    self.exchange.new_order(order)
                else:
                    sell_price = current_price + 50
                    order.price = sell_price
                    self.exchange.replace_order(order)
            if order.side == 'buy':
                if not order.active:
                    buy_price = current_price - 50
                    order.original_price = buy_price
                    order.price = buy_price
                    self.exchange.new_order(order)
                else:
                    buy_price = current_price - 50
                    order.price = buy_price
                    self.exchange.replace_order(order)

    def order_sell_handler(self, order):
        current_price = self.exchange.get_current_price(self.symbol)
        sell_price = current_price + 50
        if not order.active:
            sell_price = current_price + 50
            order.original_price = sell_price
            order.price = sell_price
            self.exchange.new_order(order)

    def order_buy_handler(self, order):
        current_price = self.exchange.get_current_price(self.symbol)
        buy_price = current_price - 50
        if not order.active:
            order.price = buy_price
            self.exchange.new_order(order)


class avg_strat_tiers(generic_limit_strat):
    def __init__(self, exchange, account, manager_id, symbol, percent_to_manage):
        super().__init__('avg_limit_strat')
        self.fucn=lambda x: x ** 2
        self.ema_price = None
        self.account = account
        #self.params['increment'] = [1.015]
        self.params['lookback'] = 5
        self.params['percent_loss'] = [0.9999, .999]
        self.params['furthest_back'] = 11
        #self.params['global_lookback'] = [100]
        self.params['buy_tiers'] = [7]
        self.params['max_buydown'] = [0.005]
        self.params['ema_periods'] = 5
        self.params['percent_up_modifier'] = 0.5
        self.list_of_amt_objs = []
        #fucn = lambda x: x ** 2
        self.current_name=None
        self.buy_tiers, self.sell_tiers = None, None
        self.exchange = exchange
        self.exchange.set_manager(manager_id)
        self.set_manager_id = manager_id
        self.symbol = symbol
        self.percent_to_manage = percent_to_manage
        self.list_of_accepted_currencies = ['USD', 'USDT', 'BTC', 'ETH']
        self.amt_to_manage = None
        # self.amt_to_manage, _, _ = self.account.amount_of_total_funds_to_handle(percent=self.percent_to_manage, time_iso=start_time_iso, end_currency=self.symbol, list_of_accepted_currencies=self.list_of_accepted_currencies)
        self.custom_id = 0
        self.final_price_and_volume = None
        # self.final_price_and_volume = self.get_info_for_tiers(now_iso_format=start_time_iso, amt_to_manage=self.amt_to_manage)
        self.list_of_amt_objs = []
        self.made_objs = False
        self.place_buy_lock = Lock()
        self.TOLERANCE = .0001
        self.testing = False
        self.use_supertrend = True
        self.simulating = False
        self.df = None

    def create_amt_objs(self):
        price = self.exchange.get_current_price(self.symbol+'USD')
        #TODO: make sure price is being recorded correctly
        hello = 'hello1'
        amount, current_price, total = self.account.amount_of_total_funds_to_handle(percent=self.percent_to_manage, time_iso=self.now_iso_format, end_currency=self.symbol, list_of_accepted_currencies=self.list_of_accepted_currencies, current_price=price)
        self.amt_to_manage = amount
        self.buy_tiers, self.sell_tiers = self.get_tiers(self.params['buy_tiers'], self.params['max_buydown'], self.fucn)
        self.final_price_and_volume = self.get_info_for_tiers(now_iso_format=self.now_iso_format, amt_to_manage=self.amt_to_manage)
        self.list_of_amt_objs = self.preliminary_create_amt_objs(number_amt_objs=self.params['buy_tiers'], symbol=self.symbol, manager_id=self.set_manager_id, final_price_and_volume = self.final_price_and_volume)

    def prepare_and_check_before_decision(self):
        amount, current_price, total = self.account.amount_of_total_funds_to_handle(percent=self.percent_to_manage, time_iso=self.now_iso_format, end_currency=self.symbol, list_of_accepted_currencies=self.list_of_accepted_currencies)
        self.amt_to_manage = amount
        self.final_price_and_volume = self.get_info_for_tiers(now_iso_format=self.now_iso_format, amt_to_manage=self.amt_to_manage)
        self.check_all_amts_against_total(amount)
        one_list, suggested_trades = self.get_all_suggested_trades_one_list()
        return one_list, suggested_trades, current_price

    def helper_make_amt_objs(self):
        for idx, amt_obj in enumerate(self.list_of_amt_objs):
            amt_obj.set_amount_to_manage_setter(self.final_price_and_volume[idx][1])
            # current_obj = get_amount(symbol=self.symbol, amount_to_manage=self.final_price_and_volume[i][1],custom_id=self.custom_id, manager_id=self.set_manager_id, additional_info=['tiers', str(i)])

    def get_all_amt_objects(self):
        return self.list_of_amt_objs

    def get_info_for_tiers(self, total_amt_to_manage, buy_tiers):
        #IMPORTANT: this assumes that self.buy_tiers is already populated...from the functioin get_tiers
        final_price_and_volume = []
        ema_average_last_element = self.get_ema(symbol=self.symbol, num_prev_candles=self.params['ema_periods'], len_period_minutes=self.params['lookback'])
        total_volume = total_amt_to_manage        
        for percent_ema_vol in buy_tiers:
            price_element = Decimal(percent_ema_vol[0])*Decimal(ema_average_last_element)
            volume_element = Decimal(percent_ema_vol[1])*Decimal(total_volume)
            final_price_and_volume.append([price_element, volume_element])
        return final_price_and_volume

    def get_list_of_suggested_trades(self):
        all_lists_of_suggested_trades = []
        total_amt = 0
        for amt_obj in self.list_of_objs:
            amt_object_suggested_trades = amt_obj.get_size_buy_sell()
            for trade in amt_object_suggested_trades:
                total_amt += Decimal(trade.total_amt)
            all_lists_of_suggested_trades.append(amt_object_suggested_trades)
        return all_lists_of_suggested_trades, total_amt



    # def check_all_amts_against_total(self, amt_allowed):
    #     total_addition = 0
    #     for idx, amt_obj in enumerate(self.list_of_amt_objs):
    #         amt_obj.set_amt_to_manage(self.final_price_and_volume[amt_obj.tier_assigned][1])
    #         total_1, obj_amt_to_manage = amt_obj.consider_amt()
    #         total_addition += obj_amt_to_manage
    #     if abs(total_addition-amt_allowed) >= self.TOLERANCE:
    #         raise ValueError('Too much or too little allocated')

    def get_all_suggested_trades_one_list(self):
        all_suggested_trades, total_amt = self.get_list_of_suggested_trades()
        suggested_trades = []
        for every_list in all_suggested_trades:
            for trade in every_list:
                suggested_trades.append(trade)
        return copy.deepcopy(suggested_trades), suggested_trades

    def run_once(self):
        amount, _, _=self.account.amount_of_total_funds_to_handle(percent=self.percent_to_manage, end_currency='BTC', list_of_accepted_currencies=self.list_of_accepted_currencies, current_price=self.exchange.get_current_price(self.symbol), manager_id = self.set_manager_id)
        self.amt_to_manage = amount
        # TODO make this function run first in sim_test or what was sim_test
        #Like arduino there is a section for things to run only once to initialize things, I think that is important here as well
        self.buy_tiers, self.sell_tiers = self.get_tiers(self.params['buy_tiers'], self.params['max_buydown'], self.fucn)
        # self.final_price_and_volume = self.get_info_for_tiers(now_iso_format=self.now_iso_format, amt_to_manage=self.amt_to_manage)
        self.final_price_and_volume = self.get_info_for_tiers(total_amt_to_manage=self.amt_to_manage, buy_tiers=self.buy_tiers)
        self.list_of_objs = self.initialize_amt_objs(number_amt_objs=self.params['buy_tiers'], symbol=self.symbol, manager_id=self.set_manager_id)
        for idx, amt_obj in enumerate(self.list_of_objs):
            amt_obj.append_additional_info(key='tier_assigned', element = idx)
            amt_obj.set_amount_to_manage_setter(self.final_price_and_volume[idx][1])
        # self.list_of_amt_objs = self.preliminary_create_amt_objs(number_amt_objs=self.params['buy_tiers'], symbol=self.symbol, manager_id=self.set_manager_id, final_price_and_volume = self.final_price_and_volume)

    def decision(self):
        self.final_price_and_volume = self.get_info_for_tiers(total_amt_to_manage=self.amt_to_manage, buy_tiers=self.buy_tiers)
        orders_copy, orders_editable = self.get_all_suggested_trades_one_list()
        self.supertrend_value = self.exchange.get_supertrend(backup=10)
        # TODO: IF there is a problem with trade consistency use orders_copy because orders_editable actually changes the produced trade automatically...if you mess with it you could cause a lot of unfoseen issues
        for order in orders_editable:
            # dictionary now?
            if order.side == 'buy':
                order.price = self.final_price_and_volume[order.additional_info['tier_assigned']][0]
                if not order.active:
                    self.exchange.new_order(order)
                else:
                    self.exchange.replace_order(order)
            elif order.side == 'sell':
                if not order.active:
                    self.ema_price = self.get_ema(symbol=self.symbol, num_prev_candles=self.params['ema_periods'], len_period_minutes=self.params['lookback'])
                    price_from_tier_if_buy = Decimal(self.final_price_and_volume[order.additional_info['tier_assigned']][0])
                    percent_modifier = Decimal(self.params['percent_up_modifier'])
                    initial_sell_price = Decimal(Decimal(self.ema_price) + price_from_tier_if_buy)  * percent_modifier
                    order.original_price=initial_sell_price
                    order.price = initial_sell_price
                    self.exchange.place_order(order)
                else:
                    new_price = self.sell_decay(first_price=order.original_price, decay_time=order.original_sell_time, rate=self.params['percent_loss'], period_length=self.params['lookback'] )
                    order.price = new_price
                    self.exchange.replace_order(order)
        
    def order_sell_handler(self, order):
        # TODO: make sure that 
        self.ema_price = self.get_ema(symbol=self.symbol, num_prev_candles=self.params['ema_periods'], len_period_minutes=self.params['lookback'])
        price_from_tier_if_buy = Decimal(self.final_price_and_volume[order.additional_info['tier_assigned']][0])
        percent_modifier = Decimal(self.params['percent_up_modifier'])
        initial_sell_price = Decimal(Decimal(self.ema_price) + price_from_tier_if_buy)  * percent_modifier
        order.original_price=initial_sell_price
        order.price = initial_sell_price
        self.exchange.new_order(order)

    def order_buy_handler(self, order):
        self.final_price_and_volume = self.get_info_for_tiers(total_amt_to_manage=self.amt_to_manage, buy_tiers=self.buy_tiers)
        order.price = self.final_price_and_volume[order.additional_info['tier_assigned']][0]
        self.exchange.new_order(order)
        # if self.use_supertrend:
        #     for idx, trade in enumerate(final_list):
        #         # if trade.price < 1  :
        #         #     pass
        #         if abs(difference) < 1:
        #             really_final.append(trade)
        #             really_final_editable.append(final_list_editable[idx])
        #         elif difference < 0 and trade.side == 'sell':
        #             really_final.append(trade)
        #             really_final_editable.append(final_list_editable[idx])
        #     if really_final is not None:
        #         final_list = really_final
        #         final_list_editable = final_list_editable
        # return final_list, final_list_editable

    def calculate_slope(self, data, supertrend_column, start_idx, end_idx):
        x = np.arange(start_idx, end_idx +1)
        y = data.iloc[start_idx:end_idx + 1][supertrend_column].values
        slope, _ = np.polyfit(x, y, 1)
        return slope

    # def get_ema(self, data_pd_format):
    #     while len(data_pd_format.index) < self.params['ema_periods'] + 1:
    #         last_line = data_pd_format.iloc[-1]
    #         data_pd_format = data_pd_format.append(last_line)
    #     averages = pd.Series(data_pd_format['open'].ewm(span=self.params['ema_periods'], min_periods=self.params['ema_periods']).mean(), name='EMA')
    #     return averages

    def get_tiers(self, tiers_on_each_side, max_percent_change, volume_funct):
        # Get individual limit tiers from the number of tiers, the maximum distance from the current price,
        #   and a function to fit the volume units
        # Returns a dictionary representing the buy/sell prices (keys) and their volumes (values)
        #   as a percentage of total volume available
        #These initial if statements should never actuate...if it does something is wrong with the Permuation class call or otherwise
        if isinstance(tiers_on_each_side, list):
            tiers_on_each_side = tiers_on_each_side[0]
        if isinstance(max_percent_change, list):
            max_percent_change = max_percent_change[0]
        volumes = [volume_funct(i) for i in range(1, tiers_on_each_side + 1)]
        # Scale volumes to be a percent
        total = sum(volumes)
        volumes = [volume / total for volume in volumes]

        percent_at_a_time = max_percent_change / tiers_on_each_side
        sell_tiers = [1.0 + percent_at_a_time * i for i in range(1, tiers_on_each_side + 1)]
        buy_tiers = [1.0 - percent_at_a_time * i for i in range(1, tiers_on_each_side + 1)]
        return list(zip(buy_tiers, volumes)), list(zip(sell_tiers, volumes))


# class avg_limit_strat(generic_limit_strat): 
#     def __init__(self):
#         super().__init__('avg_limit_strat')
#         self.params['increment'] = [1.015]
#         self.params['lookback'] = [5]
#         self.params['percent_loss'] = [0.99]
#         self.params['global_lookback'] = [100]

#     def helper_make_amt_objs(self):
#         return [True, 1]
    
#     def send_list_of_amt_objs(self, list_of_amt_objs):
#         self.list_of_amt_objs = list_of_amt_objs

#     def helper_total_periods_passed(self, date_time_object_start, date_time_object_end, period_len_minutes):
#         time_passed_delta = date_time_object_start - date_time_object_end
#         return ((abs(time_passed_delta.days) * 3600 + abs(time_passed_delta.seconds) + abs(time_passed_delta.microseconds) / 1000000)/60)//period_len_minutes

#     def sell_decay(self, trade, editable_trade, now_iso_format, current_price):
#         print('second sell')
#         parent_id_timestamp = datetime.fromisoformat(editable_trade.parent_timestamp)
#         total_periods_passed = self.helper_total_periods_passed(parent_id_timestamp, datetime.fromisoformat(now_iso_format), self.params['lookback'])
#         trade.price = trade.original_price * pow(Decimal(self.params['percent_loss']), Decimal(total_periods_passed))
#         editable_trade.last_time_modified = now_iso_format
#         editable_trade.price = trade.price
#         if self.testing:
#             trade.price = current_price + 5
#             editable_trade.price = trade.price

#     def decision(self, now_iso_format, list_of_suggested_trades, exchange, additional_info = None):
#         new_suggested_trades = []
#         total_periods_passed = 0.0
#         copy_list = copy.deepcopy(list_of_suggested_trades)
#         if not isinstance(exchange, Exchange):
#             exchange = exchange()
#         for suggested_trade in copy_list:
#             if suggested_trade.last_time_modified != "":
#                 mm = self.helper_total_periods_passed(datetime.fromisoformat(now_iso_format), datetime.fromisoformat(suggested_trade.last_time_modified), self.params['lookback'])
#             if suggested_trade.side == 'sell' and suggested_trade.price == -1:
#                 suggested_trade.price = round(suggested_trade.parent_price * self.params['increment'], 2)
#                 suggested_trade.original_price = suggested_trade.price
#                 suggested_trade.last_time_modified = now_iso_format
#                 suggested_trade.total_times_modified += 1
#             elif suggested_trade.side == 'sell' and suggested_trade.price != -1 and self.helper_total_periods_passed(datetime.fromisoformat(now_iso_format), datetime.fromisoformat(suggested_trade.last_time_modified), self.params['lookback']) >= 1:
#                 parent_id_timestamp = datetime.fromisoformat(suggested_trade.timestamp)
#                 total_periods_passed = self.helper_total_periods_passed(parent_id_timestamp, datetime.fromisoformat(now_iso_format), self.params['lookback'])
#                 suggested_trade.price = suggested_trade.original_price * pow(self.params['percent_loss'], total_periods_passed)
#                 suggested_trade.last_time_modified = now_iso_format
#             elif suggested_trade.side == 'buy' and suggested_trade.last_time_modified == '' or self.helper_total_periods_passed(datetime.fromisoformat(now_iso_format), datetime.fromisoformat(suggested_trade.last_time_modified), self.params['lookback']) >= 1:
#                 start_time_object = datetime.fromisoformat(now_iso_format) - timedelta(seconds=self.params['global_lookback'] * self.params['lookback'] * 60)
#                 start_time_iso = datetime.isoformat(start_time_object)
#                 history = exchange.get_candles_on_timestamps(symbol=suggested_trade.symbol, start_date=start_time_iso, end_date=now_iso_format, period_len_minutes=self.params['lookback'])
#                 history = pd.DataFrame.from_records(candle.to_dict() for candle in history)
#                 total = 0
#                 for i in range(len(history.index)):
#                     total += history.iloc[i]['close']
#                 buy_percent = 2.0 - self.params['increment']
#                 percent_lower = ((1.0 / buy_percent) - (1.0 / (len(history.index) + 1.0))) * (len(history.index) + 1.0)
#                 suggested_trade.price = total / percent_lower
#                 suggested_trade.last_time_modified = now_iso_format
#             new_suggested_trades.append(suggested_trade)
#         return new_suggested_trades
# class bb_limit_strat(generic_limit_strat):
#     def __init__(self):
#         super().__init__('bb_limit_strat')
#         self.params['periods'] = [20, 40, 60]

#     def decision(self, history):
#         sellout_value = history.iloc[-1]['Bollinger_T_' + str(self.params['periods'])]
#         buyout_value = history.iloc[-1]['Bollinger_B_' + str(self.params['periods'])]
#         return buyout_value, sellout_value
