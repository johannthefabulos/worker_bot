import copy
import os
import sys

from exchanges.hitbtc import hitbtc

sys.path.insert(0, os.getcwd())
import json
import os
import sys
import time
from datetime import *
from datetime import datetime
from decimal import ROUND_DOWN, Decimal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from misc.json_input import *
from render.StockTradingGraph import StockTradingGraph
from scipy import stats
from tests.Features import *
from tests.Strats import *

from exchanges.Exchange import Exchange


class stored_trade:
    def __init__(
        self, manager_id, id, parent_id, price, symbol, amt, active, side, timestamp, level
    ):
        self.manager_id = manager_id
        self.id = id
        self.parent_id = parent_id
        self.price = price
        self.symbol = symbol
        self.amt = amt
        self.active = active
        self.side = side
        self.timestamp = timestamp
        self.level = level
        self.amt_id = None

    def __eq__(self, other) -> bool:
        try:
            return (
                int(self.manager_id) == int(other.manager_id)
                and (int(self.id) == int(other.id))
                and (self.parent_id == other.parent_id)
                and (self.price == other.price)
                and (self.symbol == other.symbol)
                and (self.amt == other.amt)
                and (self.active == other.active)
                and (self.side == other.side)
                and (self.timestamp == other.timestamp)
                and (self.level == other.level)
            )
        except Exception as e:
            pass

    def set_active(self, active):
        self.active = active

    def to_dict(self):
        return {
            "manager_id": self.manager_id,
            "id": self.id,
            "parent_id": self.parent_id,
            "price": self.price,
            "symbol": self.symbol,
            "amt": self.amt,
            "active": self.active,
            "side": self.side,
            "timestamp": self.timestamp,
            "level": self.level,
        }


class suggested_trade:
    def __init__(
        self, needed_id, timestamp, total_amt, side, parent_id, manager_id, parent_price, symbol, additional_info):
        self.needed_id = needed_id
        self.timestamp = timestamp
        self.total_amt = total_amt
        self.side = side
        self.parent_id = parent_id
        self.manager_id = manager_id
        self.parent_price = parent_price
        self.symbol = symbol
        self.symbol_obj = None
        self.original_sell_time = None
        self.price = -1
        self.original_price = 0
        self.last_time_modified = ""
        self.total_times_modified = 0
        self.tier_assigned = -1
        self.additional_info = None
        self.amt_id = None
        self.order_for_add_to_path = []
        self.active = False
        self.parent_timestamp = None
        self.filled = 0
        self.traded_price = None
        self.expired = ''
        self.market = False
        self.ghost = False
        self.placed = False
        self.additional_info = additional_info
        self.manager=None
        self.assigned_id=None
        self.parent_market=False
        self.completed = False

    

    def to_dict(self):
        return {
            "needed_id": self.needed_id,
            "timestamp": self.timestamp,
            "amt": self.total_amt,
            "side": self.side,
            "original": self.orignal_price,
            "last_modified": self.last_time_modified,
        }

    def __eq__(self, other) -> bool:
        try:
            return (
                self.needed_id == other.needed_id
                and (self.timestamp == other.timestamp)
                and (self.total_amt == other.total_amt)
                and (self.side == other.side)
                and (self.level == other.level)
                and (self.parent == other.parent)
                and (self.price == self.price)
                and (self.parent_price == other.parent_price)
                and (self.original_price == other.original_price)
                and (self.last_time_modified == other.last_time_modified)
            )   
        except:
            return False

class get_amount:
    def __init__(self, symbol, amount_to_manage, custom_id, manager_id, additional_info=None, file_name=None):
        self.additional_info = {}
        self.symbol = symbol
        self.amount_to_manage = amount_to_manage
        if additional_info is not None:
            for additional_element in additional_info:
                additional_info[additional_element[0]] = additional_element[1]
        self.greatest_so_far = 0
        self.greatest_key_so_far = 0
        self.manager_id = manager_id
        self.stored_trade = None
        self.suggested_trades = []
        self.TOLERANCE = 0.000001
        self.amt_obj_id = custom_id
        self.profit = 0
        self.dict_of_trades_all = {}
        self.dict_of_recent_trades = {}
        self.needs_initial_trade = True
        self.dict_of_suggested_trades = {}
        if amount_to_manage is not None:
            initial_suggested_trade = suggested_trade(side='buy', parent_id=None, needed_id=0, timestamp=None, total_amt=amount_to_manage, manager_id=manager_id, parent_price='N/A', symbol=symbol, additional_info=additional_info)
            self.needs_initial_trade = False
            # initial_suggested_trade.tier_assigned = self.tier_assigned
            initial_suggested_trade.amt_id = self.amt_obj_id
            self.dict_of_suggested_trades = {0: initial_suggested_trade}
        self.order_next_path_round = []
        self.manager = None

    def set_manager(self, manager):
        self.manager = manager
    def set_additional_info(self, set_dict):
        self.additional_info = set_dict

    def append_additional_info(self, key, element):
        self.additional_info[key] = element

    # TODO: Noticed this is similar to another funciton in name...this should be set_amount and the other one should be named something else
    def set_amount_to_manage_setter(self, amt_to_manage):
        if self.needs_initial_trade:
            initial_suggested_trade = suggested_trade(side='buy', parent_id=None, needed_id=0, timestamp=None, total_amt=amt_to_manage, manager_id=self.manager_id, parent_price='N/A', symbol=self.symbol, additional_info=self.additional_info)
            self.needs_initial_trade = True
            initial_suggested_trade.amt_id = self.amt_obj_id
            self.dict_of_suggested_trades = {0: initial_suggested_trade}
            self.needs_initial_trade = False
        self.amount_to_manage = amt_to_manage


    def add_to_path(self, filled_order):
        """Function to deal with a filled order and set corresponding suggested order

        Args:
            filled_order (suggested_trade object): many different parameters for many different strategies
        """
        keys = self.dict_of_suggested_trades.keys()
        max_keys = max(keys)
        found = False
        # Go through our dictionary and see which one matches the filled order
        for key in keys:
            if self.dict_of_suggested_trades[key].needed_id == filled_order.needed_id:
                found = True
                held = self.dict_of_suggested_trades[key]
                # Regardless of how much was sold(if it was sold) we want to get all the profit
                if filled_order.side == 'sell':
                    temp_price_parent = Decimal(filled_order.parent_price)
                    temp_price_now = Decimal(filled_order.price)
                    amt_decimal = Decimal(filled_order.total_amt)
                    self.profit += (temp_price_now * amt_decimal) - (temp_price_parent * amt_decimal)
                # If the filled order completely covers the suggested trade we can immediately put it in dict_of_trades
                if abs(Decimal(filled_order.total_amt) - Decimal(self.dict_of_suggested_trades[key].total_amt)) < self.TOLERANCE:
                    # if filled_order.side == 'sell':
                    #     temp_price_parent = Decimal(filled_order.parent_price)
                    #     temp_price_now = Decimal(filled_order.price)
                    #     amt_decimal = Decimal(filled_order.total_amt)
                    #     self.profit += (temp_price_now * amt_decimal) - (temp_price_parent * amt_decimal)
                    #     hello = 'hello'
                    self.dict_of_suggested_trades[key].timestamp = filled_order.timestamp
                    self.dict_of_suggested_trades[key].price = filled_order.price
                    trade_obj = self.dict_of_suggested_trades.pop(key)
                    self.add_to_dict_of_recent_trades(trade_obj)
                # Otherwise we only know part of the suggested trade happened
                else:
                    # First decrement the amount of the filled order 
                    self.dict_of_suggested_trades[key].total_amt -= Decimal(filled_order.total_amt)
                    # Going to move the recent trade that happened
                    self.add_to_dict_of_recent_trades(filled_trade=filled_order)
                return self.create_new_suggested_trade(parent=held, filled_order=filled_order, max_keys= max_keys), self.dict_of_suggested_trades, self.dict_of_recent_trades
        # If the filled order doesn't match a suggested trade something has gone wrong
        if not found:
            raise ValueError("the filled order did not match a suggested order")

    def get_size_buy_sell(self):
        """Function to give all suggested trades

        Returns:
            list: list of all the suggested trades
        """
        all_suggested_trades = []
        keys = self.dict_of_suggested_trades
        # Go through every key to get all the suggested trades
        for key in keys:
            all_suggested_trades.append(self.dict_of_suggested_trades[key])
        self.consider_amt()
        return all_suggested_trades

    def consider_amt(self):
        """Function to raise error if we are out of capacity

        Raises:
            ValueError: raise error if all the combined amounts are greater the the amount to manage
        """
        all_suggested_trades = []
        total = 0
        keys = self.dict_of_suggested_trades.keys()
        # Go throught the dict of suggested trades and make sure they add up the the amount the manager is supposed to be using
        for key in keys:                                                                                  
            total += Decimal(self.dict_of_suggested_trades[key].total_amt)
        for order_to_change in self.order_next_path_round:
            difference = order_to_change[0]
            amt = order_to_change[1]
            if difference: total -= amt
            if not difference: total += amt
        if abs(total - self.amount_to_manage) > self.TOLERANCE:
            raise ValueError('Trying to trade with more than should be able to')
        return total, self.amount_to_manage

    def set_amt_to_manage(self, amt_to_manage):
        """Function to allow for a seamless transition of the amount of currecy the object is responsible for

        Args:
            amt_to_manage (float): the new amt that the object should be responsible for
        """
        # Initial set to None
        difference = None
        # If the new amount is less we need to find how much to subtract from buy orders
        if amt_to_manage < self.amount_to_manage:
            amount = self.amount_to_manage - amt_to_manage
            difference = True
        # If the new amount is more than the old one it should be pretty simple to add funds to the suggested trades
        else:
            amount = amt_to_manage - self.amount_to_manage
            difference = False
        keys = self.dict_of_suggested_trades.keys()
        self.amount_to_manage = amt_to_manage
        self.helper_amt_to_manage(difference= difference, amount_for_fucn=amount, keys=keys)
        
    def helper_amt_to_manage(self, difference, amount_for_fucn, keys):
        """Function to sort through self.dict_of_suggested_trades and find where we can add or subtract

        Args:
            difference (bool): If true it means that we want to trade with less than what we currently are
            amount_for_fucn (float): amount of currency we need to add or subtract based on difference bool
            keys (list): list of all the keys to the dictionary

        Raises:
            ValueError: difference should not be None and if it is raise an error
        """
        # Loop through all the elements in the dictionary of suggested trades
        for key in keys:
            # We only want to affect buys since we always want to sell the currency we already have
            if self.dict_of_suggested_trades[key].side == 'buy':
                # If difference is true then we want to be trading with less than we currently are
                if difference:
                    # If the suggested trade we are looking at is greater than the difference we just subtract the amount we want to buy
                    if self.dict_of_suggested_trades[key].total_amt > amount_for_fucn:
                        self.dict_of_suggested_trades[key].total_amt = Decimal(self.dict_of_suggested_trades[key].total_amt)
                        self.dict_of_suggested_trades[key].total_amt -= amount_for_fucn
                        amount_for_fucn = 0
                    # Otherwise we need to keep searching for buy orders until we get the full amount less we want to trade with
                    else:
                        amount_for_fucn -= self.dict_of_suggested_trades[key].total_amt
                        self.dict_of_suggested_trades.pop(key)
                        break
                # If difference is false all we need to do is add the extra funding to a current buy order
                elif difference == False:
                    if not isinstance(self.dict_of_suggested_trades[key].total_amt, Decimal):
                        self.dict_of_suggested_trades[key].total_amt = Decimal(self.dict_of_suggested_trades[key].total_amt)
                    self.dict_of_suggested_trades[key].total_amt += amount_for_fucn
                    amount_for_fucn = 0
                else:
                    raise ValueError('difference should not be None')
        # This is the case that there are no current buys or not enough buys to satisfy requirements
        if amount_for_fucn > 0:
            # We are going to look at this variable next time a trade is added so we can add less or more depending on what was left
            self.order_next_path_round.append([difference, amount_for_fucn])

    def add_to_dict_of_recent_trades(self, filled_trade):
        """Function to gently place trades that have occured into a different dictionary.
        Mostly a quality of life function if we ever need to acces trades that have happened already but
        we shouldn't need to

        Args:
            filled_trade (suggested_trade obj): Trade that was filled and we add to the dict of recent trades
        """
        done = False
        keys = self.dict_of_recent_trades.keys()
        if len(keys) == 0:
            self.dict_of_recent_trades[0] = filled_trade
            done = True
        # Loop through all the trades and find if one with the id exists
        if not done:
            for key in keys: 
                if self.dict_of_recent_trades[key].needed_id == filled_trade.needed_id:
                    #If it exists then we merely add the amount to what is already there
                    if not isinstance(self.dict_of_recent_trades[key].total_amt, Decimal):
                        self.dict_of_recent_trades[key].total_amt = Decimal(self.dict_of_recent_trades[key].total_amt)
                    self.dict_of_recent_trades[key].total_amt += Decimal(filled_trade.total_amt)
                    done = True
        # If done is not true then we know we know we have to create a new dictionary key for the filled trade
        if not done:
            max_keys = max(keys)
            self.dict_of_recent_trades[max_keys+1] = filled_trade

    def get_opposite(self, side):
        """Simple functiont to return opposite of the side of a trade

        Args:
            side (string): 'buy' or 'sell'

        Returns:
            string: opposite side of the argument
        """
        return 'sell' if side == 'buy' else 'buy'
    def check_if_combineable(self):
        is_combineable = False
        all_keys = list(self.dict_of_suggested_trades.keys())
        new_dict = {}
        for key in all_keys:
            current_suggested = self.dict_of_suggested_trades[key]
            new_dict[current_suggested.parent_id] = []
            if current_suggested.parent_market:
                new_dict[current_suggested.parent_id].append(current_suggested)
        filtered_dict = {key: value for key, value in new_dict.items() if value}
        filtered_keys = list(filtered_dict.keys())
        total_combined = 0
        needed_ids = []
        time_to_break = False
        for key in filtered_keys:
            if len(filtered_dict[key]) > 1:
                for idx, element in enumerate(filtered_dict[key]):
                    if idx == len(filtered_dict[idx]) - 1:
                        element.total_amt += total_combined
                        time_to_break = True
                        break
                    else:
                        total_combined += element.total_amt
                        needed_ids.append(element.needed_id)
            if time_to_break:
                for needed_id in needed_ids:
                    self.dict_of_suggested_trades.pop(str(key))
                break

    def create_new_suggested_trade(self, parent, filled_order, max_keys):
        """Function to make sure that we add corresponding order in light of a suggested trade being filled
        and also to check that if the amount to manage has been changed that a buy will either not occur
        buy for a greater amount than suggested or buy for less than the amount suggested

        Args:
            parent (suggested_trade obj): suggested trade as reasoned by get_amount class
            filled_order (suggested_trade): this order has been filled and needs a corresponding order if there was no change self.to amt_to_manage
            max_keys (int): the max key amount so we can assign a new dictionary trade
        """
        # Generally we wish to proceed with creating a new dictionary element
        proceed = True
        # If there is an order to increase or reduce funds because there were no buys before we need to address the change in amount to manage here
        if self.get_opposite(parent.side) == 'buy':
            # Go through all the orders to add or subtract funds from the add_to_path fucn
            for idx, params in enumerate(self.order_next_path_round):
                modified_difference = params[0]
                modified_amt = params[1]
                # If the order was reduced funding and the amount is less than the filled_order_amt we can simply subtract to satisfy the order
                if modified_difference and modified_amt < filled_order.total_amt:
                    if not isinstance(filled_order.total_amt, Decimal):
                        filled_order.total_amt = Decimal(filled_order.total_amt)
                    filled_order.total_amt -= Decimal(modified_amt)
                    # Since the order is satisfied we delete it from the list
                    self.order_next_path_round.pop(idx)
                # If the order was reduced funding and the filled order doesn't have enough to satisfy it then we won't proceed with the buy order
                if modified_difference and modified_amt > filled_order.total_amt:
                    proceed=False
                    # The order to reduce funding will have to be addressed by a different buy in the future
                    params[1]-=filled_order.total_amt
                    break
                # If the order is to increase funding then it is easy to simply increase the suggested trade buy amount
                if not modified_difference:
                    if not isinstance(filled_order.total_amt, Decimal):
                        filled_order.total_amt = Decimal(filled_order.total_amt)
                    filled_order.total_amt+=modified_amt
                    # Since the order to increase was satisfied we remove it
                    self.order_next_path_round.pop(idx)
        # Assuming there wasn't an order that wiped out the need for a buy we can now add the corresponding suggested trade to the dictionary
        if proceed:
            self.greatest_so_far += 1
            new_suggested_trade = suggested_trade(needed_id=self.greatest_so_far, 
                                                 timestamp=None, 
                                                 total_amt=filled_order.total_amt, 
                                                 side=self.get_opposite(parent.side),
                                                 parent_id=parent.needed_id,
                                                 manager_id=self.manager_id,
                                                 parent_price=parent.price,
                                                 symbol=parent.symbol,
                                                 additional_info=self.additional_info)
            new_suggested_trade.amt_id = parent.amt_id
            new_suggested_trade.parent_market = parent.market
            new_suggested_trade.price = -1
            new_suggested_trade.parent_timestamp = parent.timestamp
            self.dict_of_suggested_trades[max_keys + 1] = new_suggested_trade
            # self.check_if_combineable()
            new_suggested_trade = self.dict_of_suggested_trades[max_keys + 1]
            return new_suggested_trade


class manage_trades:
    def __init__(self, strat, exchange, manager_id, symbol, percent_to_manage):
        self.strat = strat
        self.exchange = exchange
        self.manager_id = manager_id
        self.symbol = symbol
        self.percent_to_manage = percent_to_manage
        self.all_amount_objects = []
        """
        if isinstance(data_to_use, pd.DataFrame):
            self.data_to_use = data_to_use
        else:
            self.data_to_use = None
        """
        #self.amt_obj = get_amount(self.symbol)
        self.active = True
        self.period_minutes_size = 0
        if not isinstance(self.exchange, Exchange):
            self.exchange = exchange()
        #self.reclaim_data("stored_tradess.csv")

    def __eq__(self, other) -> bool:
        return (
            (self.manager_id == other.manager_id)
            and (self.limit_strat.name == other.limit_strat.name)
            and (self.exchange.exchange_name == other.exchange.exchange_name)
            and (self.amount_to_manage == other.amount_to_manage)
        )

    def get_all_current_orders(self):
        all_suggested = []
        for amt_obj in self.all_amount_objects:
            amt_object_suggested_trades = amt_obj.get_size_buy_sell()
            all_suggested.append(amt_object_suggested_trades)
        return all_suggested

    def check_if_combineable(self):
        for amt_obj in self.all_amount_objects:
            amt_obj.check_if_combineable()

    def add_all_amt_objs(self):
        total_used = 0
        for amt_obj in self.all_amount_objects:
            amt_object_suggested_trades = amt_obj.get_size_buy_sell()
            for trade in amt_object_suggested_trades:
                total_used += trade.total_amt
        return total_used

    def get_strat(self):
        return self.limit_strat.name
    
    def append_amt_objects(self, amt_objects):
        self.all_amount_objects = [amt_obj for amt_obj in amt_objects]

    def get_buy_and_sell(self):
        for amt_obj in self.all_amount_objects:
            amt_obj.get_size_buy_sell(self.manager_id)

    def get_all_paths(self):
        all_paths = []
        for amt_obj in self.all_amount_objects:
            all_paths.append(amt_obj.get_path())
        return all_paths

    def get_all_profit(self):
        all_profit = []
        for amt_obj in self.all_amount_objects:
            all_profit.append([amt_obj, amt_obj.get_profit()])
            
    def add_to_path(self, filled_order):
        for amt_obj in self.all_amount_objects:
            if amt_obj.amt_obj_id == filled_order.amt_id:
                new_order = amt_obj.add_to_path(filled_order)
                return new_order

    def reclaim_data(self, file_name):
        f = open(file_name)
        all_lines = f.readlines()
        if all_lines != [] and all_lines != ["\n"]:
            trades_df = pd.read_csv(file_name, index_col=0)
            final_list = []
            if trades_df is not None:
                for i in range(len(trades_df.index)):
                    if trades_df.iloc[i]["manager_id"] == self.manager_id:
                        old_trade = stored_trade(
                            trades_df.iloc[i]["manager_id"],
                            trades_df.iloc[i]["id"],
                            trades_df.iloc[i]["parent_id"],
                            trades_df.iloc[i]["price"],
                            trades_df.iloc[i]["symbol"],
                            trades_df.iloc[i]["amt"],
                            trades_df.iloc[i]["active"],
                            trades_df.iloc[i]["side"],
                            trades_df.iloc[i]["timestamp"],
                            trades_df.iloc[i]["level"],
                        )
                        final_list.append(old_trade)
                self.amt_obj.add_list_to_path(final_list)
                return self.amt_obj.get_path()

    def to_dict(self):
        return {
            "limit_strat": self.limit_strat.name,
            "exchange": self.exchange.exchange_name,
            "amount_to_manage": self.amount_to_manage,
            "manager_id": self.manager_id,
            "active": self.active,
            "symbol": self.symbol
        }

    def return_exchange(self):
        return self.exchange.exchange_name
"""
test_strat = avg_limit_strat()
file_name = 'LTCUSDreadme.csv'
data=None
# data = pd.read_csv(file_name)
manager = manage_trades(limit_strat=test_strat,exchange=binanceus,amt_to_manage = 5,manager_id = 1,data_to_use=data)
name = manager.return_exchange()
name1 = manager.get_strat()
print(name1)
print(name)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['close'],name = 'close'))
fig.add_trace(go.Scatter(x=data['timestamp'],y = all_buys,name = 'buys'))
fig.add_trace(go.Scatter(x=data['timestamp'],y = all_sells, name = 'sells'))
fig.write_image("fig1.png")

#plt.plot(render_obj)
#plt.savefig('hello' + manage_trades.test() + '.png')
print(datetime.now())
print(manager.test())
"""
