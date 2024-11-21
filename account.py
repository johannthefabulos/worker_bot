from datetime import datetime
from inspect import isclass

import pandas as pd
import pytz
import threading
import tests.limit_strats as strats_file
from exchanges.binance import *
from exchanges.binanceus import *
from exchanges.binanceus import binanceus
from exchanges.Exchange import Exchange
from exchanges.fake_exchange import FakeExchange
from manage.managetrades import manage_trades
from tests.limit_strats import *


class Account():
    def __init__(self, dict_of_amounts, exchange=None, simulating=True):
        self.dict_of_amounts = dict_of_amounts
        self.managers = None
        self.all_strats = ['avg_limit_strat']
        self.all_exchanges = ['BINANCEUS']
        self.limit=None
        self.exchange = exchange
        self.simulating = simulating
        self.counter = 10
        self.counter_amount = 10

    def create_manager(self, strat, exchange, manager_id, symbol, percent_to_manage):
        new_obj = manage_trades(strat, exchange, manager_id, symbol, percent_to_manage)
        return new_obj
    
    def set_managers(self, managers):
        self.managers = managers
        hello = 'hello'

    def get_all_managers(self):
        return self.managers

    def record(self, filled_order):
        """Function to record to the account when we have bought or sold

        Args:
            filled_order (suggested trade object): many different variables for many different strategies
        """
        # Set our currencies and commodities based on symbol of the order
        #TODO: not sure if this function is expensive for the real exchange or not
        # there are better ways of doing this
        # keep an eye on it when doing real exchange things 
        self.dict_of_amounts = self.exchange.get_balance()
        symbol = filled_order.symbol
        if filled_order.needed_id == 23:
            print('wut')
        if 'USDT' in filled_order.symbol:
            index_seperator = symbol.index('USDT')
            currency = symbol[:index_seperator]
            commodity = symbol[index_seperator:]
        if 'USD' in filled_order.symbol:
            index_seperator = symbol.index('USD')
            currency = symbol[:index_seperator]
            commodity = symbol[index_seperator:]
        if commodity == 'USDT' and self.dict_of_amounts['USD'] > 0 :
            self.dict_of_amounts['USDT'] = self.dict_of_amounts['USD']
            self.dict_of_amounts['USD'] = 0

        # If its a buy we need to subtract commodity and add currency, otherwise we need to remove currency and add commodity
        if filled_order.side == 'buy': 
            commodity_subtracted = round(Decimal((Decimal(filled_order.total_amt) * Decimal(filled_order.price))), 6)
            currency_added = round(Decimal(filled_order.total_amt), 6)
            self.dict_of_amounts[commodity] -= commodity_subtracted
            self.dict_of_amounts[currency] += currency_added
        if filled_order.side == 'sell': 
            currency_subtracted = round(Decimal(filled_order.total_amt), 6)
            commodity_added = round((Decimal(filled_order.total_amt) * Decimal(filled_order.price)), 6)
            self.dict_of_amounts[currency] -= currency_subtracted
            self.dict_of_amounts[commodity] += commodity_added
        self.dict_of_amounts[currency] = round(self.dict_of_amounts[currency], 6)
        if self.dict_of_amounts[currency] < 0:
            self.dict_of_amounts[currency] = Decimal(0.0)
        # Make sure we never use more than what we have to work with
        self.check_no_negative_quantities(currency=currency, commodity=commodity)
        # TODO: look at comment below after finished
        # Every so often we want to rebalance things...might make this every filled order...we shall see
        if self.counter == 0:
            self.write_amt_from_manager(filled_order=filled_order)
            self.counter = self.counter_amount
        else:
            self.counter -= 1

    def get_correct_manager(self, manager_id):
        current_manager = None
        current_manager = manager_id
        if current_manager is None:
            raise ValueError('Manager not found: something went wrong here')
        return current_manager

    def write_amt_from_manager(self, filled_order):
        manager_id = filled_order.manager_id
        # manager = self.get_correct_manager(manager_id=manager_id_trade)
        percent_to_manage = manager_id.strat.percent_to_manage
        all_addition = manager_id.add_all_amt_objs()
        current_price = self.exchange.get_current_price(filled_order.symbol)
        #TODO: this feels awful accessing limit_strat like that ewwwwwwwwwwwww make this better
        # Maybe I can run it every time in decision?? too hard for user?
        amount, _, _=self.amount_of_total_funds_to_handle(percent=manager_id.strat.percent_to_manage,end_currency='BTC', list_of_accepted_currencies=manager_id.strat.list_of_accepted_currencies, current_price=current_price, manager_id=filled_order.manager_id)
        manager_id.strat.amt_to_manage = amount

    def check_no_negative_quantities(self, currency, commodity):
        self.dict_of_amounts = self.exchange.get_balance()
        """Function to make sure we never use more than what we have to work with

        Args:
            currency (string): EX 'BTC'
            commodity (string): EX 'USD'

        Raises:
            ValueError: Point of the function is to raise an error if we ever have negative quanity for currency or commodity
        """
        #if self.dict_of_amounts[currency] > self.limit:
            #raise ValueError('Should not be more than limit')
        # Quickly check if we have a negative balance and if we do raise valuerror
        if self.dict_of_amounts[currency] < -.001 or self.dict_of_amounts[commodity] < 0:
            raise ValueError('cannot have negative balance')

    def find_total_usd(self, time_iso, current_price_dict):
        total = 0
        # If we are simulating we want to get all the keys in the dict
        keys = self.dict_of_amounts.keys()
        used_price = 0
        # Go through the dictionary and key by key
        for key in keys:
            # If the key is usd or usdt then we can add the total directly
            if key == 'USD' or key == 'USDT' or key == 'USDC':
                total += Decimal(self.dict_of_amounts[key])
            # Else we need to dig through a file of trades to find what the price was at the time given in the function
            else:
                symbol = key + 'USDT'
                # line_bytes = self.helper_binary_find_target_time_iso(my_file=symbol+'.csv', target_date_obj=time_iso, after=True)
                # line_bytes holds information about the trade that occured
                total += Decimal(Decimal(current_price_dict[symbol]) * Decimal(self.dict_of_amounts[key]))
                # print('total: ', total)
                # if line_bytes[0][3] == 'BTCUSD':
                    # saved_price = line_bytes[0][4]
        # The available is the percent times the total
        # current_price_btc = Decimal(current_price_dict[symbol])
        return total

    def amount_of_total_funds_to_handle(self, percent, end_currency, list_of_accepted_currencies, current_price, manager_id):
        """When we are not simulating we need to calculate funds a little differently

        Args:
            percent (float): Percent of total USD value to work with ex. .5 for 50 of total cash in the account
            end_currency (string): the currency that we want to give an amount for 
            list_of_accepted_currencies (list of string): Sometimes there are a lot of alt currencies that we want to exclude from evaluations

        Returns:
            float, float, float: amount of end currency we can support, the price that we calculated it for, and the total funds available that was calculated from all the balances
        """
        self.exchange = manager_id.exchange
        self.dict_of_amounts = self.exchange.get_balance()
        all_keys = self.dict_of_amounts.keys()
        total = Decimal(0)
        # Go through all the elements in the dictionary
        for key in all_keys:
            # Check if the key is in the list of accepted currencies
            # We do this because sometimes we don't want to include a bunch of alt currencies in our evaluation
            if key in list_of_accepted_currencies:
                # Get the price of the last trade on the exchange in usd
                # Increment the total amount in USD if the key is either usd or usdt
                if key == 'USD' or key == 'USDT' or key == 'USDC':
                    total += Decimal(self.dict_of_amounts[key]['total'])
                # Increment the total by the price times the volume if the key is not usd
                else:
                    total += Decimal(current_price) * Decimal(self.dict_of_amounts[key]['total'])
        # The current price of the end currency which is whatever is before USD or USDT because we don't want commodity when finding prices
        # The amount of end currency is the total cash value of account times the percent we want and then divided by the last recorded price of the end currency
        amount = (total * Decimal(percent))/Decimal(current_price)
        return amount, current_price, total

    def refresh_compared_list(self):
        self.all_strats = [x for x in dir(strats_file) if isclass(getattr(strats_file, x))]
        return self.all_strats

    def reclaim_managers(self, file_location):
        managers_df = pd.read_csv(file_location,index_col=0)
        #making dataframe to read from
        self.refresh_compared_list()
        amount = len(managers_df.index)
        name_of_strat = managers_df.iloc[0]['limit_strat']
        name_of_exchange = managers_df.iloc[0]['exchange']
        for i in range(len(managers_df.index)):
            name_of_exchange = managers_df.iloc[0]['exchange']
            if managers_df.iloc[i]['limit_strat'] not in self.all_strats:
                raise Exception("no strategty with that name in managers file location")
            if name_of_exchange not in self.all_exchanges:
                raise Exception("no exchange with that name in comparison list")
            klass = globals()[name_of_strat]
            instance = klass()
            instance1 = None
            #with more exchanges make a dictiory
            if name_of_exchange == 'BINANCEUS':
                instance1 = binanceus

            self.create_manager(instance,instance1,managers_df.iloc[i]['amount_to_manage'],managers_df.iloc[i]['manager_id'])
        return self.managers