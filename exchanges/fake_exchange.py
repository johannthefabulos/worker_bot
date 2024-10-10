import copy
import os
from datetime import datetime, timedelta

import pandas as pd
import pytz

from exchanges.binanceus import binanceus
from exchanges.Exchange import Candle, Exchange
from manage.managetrades import get_amount, manage_trades, stored_trade, suggested_trade
from misc.json_input import *
from render.StockTradingGraph import StockTradingGraph
from tests.Features import *
from tests.limit_strats import *
from tests.Strats import avg_strat
from fake_server import FakeServer
#from datetime import datetime, timedelta

class trade_line:
    def __init__(self, line):
        self.price = float(line[4])
        self.volume= float(line[6].replace('/n',''))
        self.timestamp = line[2]
        self.symbol = line[3]
        self.amt_full = 0
        expiration_time = ''

class FakeExchange(Exchange):
    def __init__(self, account, start_time_iso, end_time_iso, trigger_event=None, trigger_queue=None, trigger_id = None):
        self.used_account = account
        self.amt_obj = None
        self.server_obj = FakeServer(exchange = self, start_time_iso=start_time_iso,end_time_iso=end_time_iso, used_account=self.used_account, trigger_event=trigger_event, trigger_queue=trigger_queue, trigger_id = trigger_id)

    def isolate_obj_trades(self, all_trades_w_prices, amt_obj):
        amt_obj_trades = []
        for trade in all_trades_w_prices:
            if trade.amt_id == amt_obj.amt_obj_id:
                amt_obj_trades.append(trade)
        return amt_obj_trades

    def set_manager(self, manager):
        self.server_obj.set_manager(manager)

    def set_sim_dfs(self, start_time_iso, candles_df, dict_candles_dfs):
        self.server_obj.set_sim_dfs(start_time_iso, candles_df, dict_candles_dfs)

    def check_if_skippable(self, list_of_trades, current_candle):
        skip = False
        number_found = 0
        for order in list_of_trades:
            if (order.side == 'sell' and order.price > current_candle.high) or (order.side == 'buy' and order.price < current_candle.low):
                skip = True
                number_found+=1
        if skip and number_found == len(list_of_trades):
            return True
        else:
            return False    

    def check_for_expiration(self, all_undone_objects, trade_obj, amt_obj, order):
        reset = False
        elements_to_break = []
        skip_order = False
        for idx, element in enumerate(all_undone_objects):
            if datetime.fromisoformat(trade_obj.timestamp) > element.expired:
                element.total_amt = element.filled
                amt_obj.add_to_path(element)
                self.reference_account(element)
                elements_to_break.append(idx)
                reset = True
                skip_order = True
            elif element.needed_id == order.needed_id:
                if element.filled + trade_obj.volume >= element.total_amt:
                    element.timestamp = trade_obj.timestamp
                    order.timestamp = trade_obj.timestamp
                    amt_obj.add_to_path(order)
                    self.reference_account(order)
                    elements_to_break.append(idx)
                    skip_order = True
                    reset = True
                elif element.filled + trade_obj.volume < element.total_amt:
                    element.filled += trade_obj.volume
                    element.timestamp = trade_obj.timestamp
                    skip_order = True
        for element in elements_to_break:
            try:
                all_undone_objects.pop(element)
            except:
                pass
        return reset, skip_order

    def new_order(self, order):
        self.server_obj.make_order_active(order=order)

    def replace_order(self, order):
        self.server_obj.replace_order(order=order)

    def periods_passed_now_timestamp(self, timestamp_iso, period_len_minutes):
        return self.server_obj.periods_passed_now_timestamp(timestamp_iso=timestamp_iso, period_len_minutes=period_len_minutes)

    def get_new_orders(self, current_candle, amt_obj, manager):
        list_of_suggested_trades_w_prices, list_of_suggested_trades_editable = manager.limit_strat.decision(current_candle.timestamp, amt_obj=amt_obj)
        return list_of_suggested_trades_w_prices

    def convert_locations_to_trades(self, symbol, all_trades_locations):
        print('all_trades_locations: ', all_trades_locations)
        f = open(symbol + '.csv')
        all_trades = []
        for location in all_trades_locations:
            if location == '':
                break
            f.seek(int(location))
            my_line = f.readline().split(',')
            trade_obj = trade_line(my_line)
            all_trades.append(trade_obj)
        return all_trades

    def determine_valid_candle(self, current_candle, all_trades, start_time_obj=None):
        for current_order in all_trades:
            if ((current_order.side == 'sell' and current_order.price > current_candle.high) or (current_order.side == 'buy' and current_order.price < current_candle.low)):
                return False
            elif start_time_obj and current_candle.timestamp >= start_time_obj:
                return True
            elif start_time_obj:
                return False
            elif not start_time_obj:
                return True
        return False

    def get_relevant_data(self, symbol, all_trades, candle, candles_dict, start_time_obj):
        candles_keys = list(candles_dict.keys())
        i = 0
        indicies_to_watch = []
        temp_indicies = []
        symbol = candle['symbol']
        temp_indicies = [[candle.name]]
        num_valid_candle = 0
        while 1:
            if len(temp_indicies) > 0:
                names = temp_indicies[0]
                temp_indicies.pop(0)
            else:
                break
            for name in names:
                if len(candles_keys) - 1 == i:
                    current_key = candles_keys[i]
                else:
                    current_key = candles_keys[i]
                    i+=1
                current_candle_df = candles_dict[current_key]
                line = current_candle_df.iloc[name]
                current_candle = Candle(symbol=symbol,timestamp=line['timestamp'],high=line['high'], low=line['low'],first=line['open'], last=line['close'])
                if current_key != 1:
                    valid_candle = self.determine_valid_candle(current_candle, all_trades)
                    if valid_candle:
                        num_valid_candle+=1
                        temp_indicies.append(current_candle_df.iloc[name]['step_below'])
                        break
                    if not valid_candle:
                        continue
                    current_candle_df.iloc[name]['step_below']
                else:
                    valid_candle = self.determine_valid_candle(current_candle, all_trades, start_time_obj=start_time_obj)
                    if valid_candle:
                        print('candle_df: ', current_candle_df)
                        return current_candle_df.iloc[name]['bytes']
        return None

    def get_supertrend(self, backup):
        supertrend_digit = self.server_obj.get_supertrend(backup)
        return supertrend_digit

    def get_last_candles(self, symbol_argument, num_prev_candles, len_period_minutes):
        last_candles = self.server_obj.get_last_candles(symbol_argument, num_prev_candles, len_period_minutes)
        print(last_candles)
        return last_candles
    
    def set_dict(self, my_Dict):
        self.server_obj.set_dict(my_Dict)

    def get_balance(self):
        return self.server_obj.get_balance()

    def get_current_price(self, symbol):
        current_price = self.server_obj.get_current_price(symbol)
        return current_price
    
    def get_current_time(self, order):
        return self.server_obj.get_current_time()

    def market_order(self, order):
        self.server_obj.market_order(order)

    def start(self):
        self.server_obj.simulator_control()

    def helper_find_order_that_meets_trade(self, order, bytes_in_candle, file_name, wait_time_to_complete_order):
        """Function to find the first trade in the history that meets the criteria of the order. If it doesn't exist the returns -1

        Args:
            order (suggested_trade_object): many paramaters of which different strategies use different paramaters
            bytes_in_candle (int): bytes of the trades in a associated function are recorded in a previous function
            file_name (string): file to look for the trade
            wait_time_to_complete_order (int): some trades are very small and aren't enough to fill the entire order so we can wait if we ant to
        """
        filled_order = -1
        f = open(file_name, 'r')
        start_bytes = bytes_in_candle[0]
        end_bytes = bytes_in_candle[-1]
        temp_start = copy.copy(start_bytes)
        total_volume = 0
        f.seek(temp_start)
        # Going through all the lines of the candle that we think meets the criterion of theorder
        while temp_start <= end_bytes:
            line = f.readline().split(',')
            trade_price = float(line[4])
            trade_volume = float(line[6].replace('/n',''))
            timestamp_trade = line[2]
            symbol = line[3]
            time_obj = datetime.fromisoformat(timestamp_trade)
            # Check that the criterion is reached
            if (order.side == 'sell' and order.price <= trade_price) or (order.side == 'buy' and order.price >= trade_price):
                # If the trade volume is less than the order amt then we know that we haven't filled
                if trade_volume < order.total_amt:
                    total_volume += trade_volume
                    # If we want to wait then we loop over the time we have available to see if we can fill the order
                    if wait_time_to_complete_order is not None:
                        end_time_obj = time_obj + timedelta(seconds=wait_time_to_complete_order*60)
                        while time_obj <= end_time_obj:
                            line = f.readline()
                            timestamp_trade = line[2]
                            trade_price = float(line[4])
                            if (order.side == 'sell' and order.price <= trade_price) or (order.side == 'buy' and order.price >= trade_price):
                                total_volume += float(line[6].replace('/n',''))
                            if total_volume >= order.total_amt:
                                filled_order = self.create_filled_order(order, symbol, order.total_amt, timestamp_trade)
                                return filled_order
                            time_obj = datetime.fromisoformat(line[2])
                        # Went through while while loop and didn't fill order so we make one with what we got
                        filled_order = self.create_filled_order(order, symbol, total_volume, timestamp_trade)
                        return filled_order
                else:
                    # Make order for full amount                        
                    filled_order = self.create_filled_order(order, symbol, order.total_amt, timestamp_trade)
                    return filled_order
            temp_start = f.tell()
        return filled_order

    def create_filled_order(self, order, symbol, amt, timestamp):
        """Simple function to create order

        Args:
            order (list_of_suggested_trades_obj): many different parameters for many different strategies
            symbol (string): symbol to tag to the stored trade
            amt (float): volume
            timestamp (string): iso format of the last trade to happen

        Returns:
            stored trade object: stored trade object for the get amount object to easily obtain
        """

        filled_order = order
        filled_order.timestamp = timestamp
        self.reference_account(filled_order)
        return filled_order

    def reference_account(self, filled_order):
        """Function to reverence the account associated with the exchange

        Args:
            filled_order (suggested_trade_obj): many different parameters for many different strategies
        """
        self.account.record(filled_order)

    def find_first_trade_that_meets_order(self, order, candles, byte_locations, file_name, wait_time_to_complete_order_minutes=None):
        """Function to find the first trade that meets criterion for order

        Args:
            order (suggested_trade object): Many different parameters for many different strategies
            candles (list of Candle object): convenient way for open high low close
            byte_locations (list of list of in): byte locations in the file of the individual trades for a candle
            period_size_minutes (int): length of the period size
            file_name (string): file name the byte locations correspond to
            wait_time_to_complete_order_minutes (int, optional): if we don't want to have a bunch of small trades we can wait for some of them to be filled. Defaults to None.

        Returns:
            stored trade object: stored trades are the object of get_amount to organize all the trades
        """
        filled_order = -1
        # For each candle decide if it meets the criterion for the order so we don't look at potentially thousands of trades we don't need
        for idx, candle in enumerate(candles):
            if (order.side == 'sell' and order.price <= candle.high) or (order.side == 'buy' and order.price >= candle.low):
                # If we find the trade we want we are done and can break
                filled_order = self.helper_find_order_that_meets_trade(order=order, bytes_in_candle=byte_locations[idx], file_name=file_name, wait_time_to_complete_order=wait_time_to_complete_order_minutes)
                if filled_order != -1:
                    break
        return filled_order

    def helper_create_candles(self, symbol, list_of_list_of_trades, candle_ranges):
        """Function to get open low high close of trades that happened in candle ranges

        Args:
            symbol (string): symbol that will put to every candle
            list_of_list_of_trades (list of list of list): for every candle range for every candle a list of information
            candle_ranges (list of datetime obj): normalized datetime range

        Raises:
            ValueError: value error if the first candle doesn't have a trade 

        Returns:
            list of candle objects: list of candle objects
        """
        low = 100000
        high = -1
        the_open = -1
        the_close = -1
        timestamp = ''
        # The first candle range should always have a trade associated with it
        if list_of_list_of_trades[0] is None:
            raise ValueError('The first trade accessed should always exist')
        all_candles = []
        for idx_list, list_of_trades in enumerate(list_of_list_of_trades):
            for idx, trade in enumerate(list_of_trades):
                # For each trade look through the list of information and establish open low high close from the trades
                trade[4] = float(trade[4])
                if trade[4] < low: low = trade[4]
                if trade[4] > high: high = trade[4]
                if idx == 0: the_open = trade[4]; timestamp = trade[2]
                if idx == (len(list_of_trades) - 1): the_close = trade[4]
            # If there were no trades then we need to make sure open and close were the same as the last candles with updated timestamp
            if low != 100000 and high != -1 and the_open != -1 and the_close != -1 and timestamp != '':
                candle = Candle(symbol=symbol, timestamp=datetime.isoformat(candle_ranges[idx_list]), first=the_open, last=the_close, low=low, high=high)
                previous_candle_info = candle
            elif previous_candle_info is not None:
                candle = Candle(previous_candle_info.symbol, timestamp=datetime.isoformat(candle_ranges[idx_list]), first=previous_candle_info.first, last=previous_candle_info.last)
                previous_candle_info = candle
                previous_candle_info = candle
            all_candles.append(candle)
            low = 100000
            high = -1
            the_open = -1
            the_close = -1
            timestamp = ''
        return all_candles

    def helper_binary_find_target_time_iso(self, file, target_date_obj, after=None):
        """Function to binary search through the file

        Args:
            file (string): file to search through
            target (string): time_iso_format to look for in the file
            after (bool, optional): Since the exact time may not exist it gets the trade that happened after the target time(True) or the first trade before it(False). Defaults to None.
        """
        f = open(file, 'r')
        file_size = os.fstat(f.fileno()).st_size
        left = 0
        right = file_size
        compare_me = None
        target = target_date_obj
        # Generic binary search adapted to bytes
        while left < right:
            mid = (left + right) // 2
            f.seek(mid)
            while f.tell() > 0 and f.read(1) != '\n':
                f.seek(f.tell() - 2, os.SEEK_SET)
            capture_f_position = f.tell()
            line = f.readline().split(',')
            while f.read(1) != '\n' and f.tell() > 0:
                f.seek(f.tell() - 2, os.SEEK_SET)
            capture_f_position1 = f.tell()
            line1 = f.readline().split(',')
            compare_me = datetime.fromisoformat(line[2])
            compare_me1 = datetime.fromisoformat(line1[2])
            if compare_me <= target and compare_me1 >= target:
                if after is True:
                    return [line1, capture_f_position1]
                else:
                    return [line, capture_f_position]
            elif target > compare_me and target > compare_me1:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    def get_range_for_candles(self, start_time_iso, end_time_iso, period_len_minutes):
        """function to get normalized range of candle times

        Args:
            start_time_iso (string): isoformat of the first trade will petal back until it reaches a mod level 
            end_time_iso (string): isoformat time, will pedal forward until reaches mode level
            period_len_minutes (int): period length in minutes so it can find the appropriate range

        Returns:
            list of datetime objects: datetime objects list
        """
        start_obj = datetime.fromisoformat(start_time_iso)
        end_obj = datetime.fromisoformat(end_time_iso)
        time_range = timedelta(seconds=60 * period_len_minutes)
        start_minute = start_obj.minute
        end_minute = end_obj.minute
        end_hour = None
        list_of_ranges = []
        # Pedal forward or backwards until we reach a standard minute deterimined by mod of the period length
        while (start_minute % period_len_minutes != 0): start_minute -= 1
        while (end_minute % period_len_minutes != 0): end_minute += 1
        # Make objects to start the following while loop to interate over
        if end_minute == 60:
            end_hour = end_obj.hour + 1
            end_minute = 0
        temp_start_obj = datetime(start_obj.year, start_obj.month, start_obj.day, start_obj.hour, start_minute, 0, 0, pytz.UTC)
        if end_hour is not None:
            temp_end_obj = datetime(end_obj.year, end_obj.month, end_obj.day, end_hour, end_minute, 0, 0, pytz.UTC)
        # Assumadly we will hit temp_end_obj since it is evenly divisble by the period_len_minutes
        else:
            temp_end_obj = datetime(end_obj.year, end_obj.month, end_obj.day, end_obj.hour, end_minute, 0, 0, pytz.UTC)
        while temp_start_obj <= temp_end_obj:
            # print(temp_start_obj, temp_end_obj)
            list_of_ranges.append(temp_start_obj)
            temp_start_obj += time_range
        return list_of_ranges

    def helper_loop_through_file(self, file_name, byte_start, byte_end, candle_ranges):
        """Function to assign each trade in a byte range of a file to the candle ranges that are normalized

        Args:
            file_name (string): filename the below start byte and end byte correspond to
            byte_start (intq): byte number of the beginning trade to sort
            byte_end (int): end byte of the start of the end line
            candle_ranges (list of datetime objects): normalized ranges to sort through

        Returns:
            list of list: list of lists of trades that occured to the corresponding candle ranges index
        """
        byte_locations = []
        trades_per_period_length = []
        # Make sure our return list is the same length of the candle_ranges list because their indexes correspond
        while len(trades_per_period_length) < len(candle_ranges): trades_per_period_length.append([]); byte_locations.append([])
        f = open(file_name, 'r')
        f.seek(byte_start)
        candle_start = 0
        byte_copy = copy.copy(byte_start)
        # Readline increments the file byte count automatically so we simply need to reset it afterword
        while byte_copy <= byte_end:
            line = f.readline().split(',')
            timestamp = datetime.fromisoformat(line[2])
            # For each line we want to assign it to one of the candle ranges, if its empty a future function assignes the open low high close of the function
            for i in range(candle_start, len(candle_ranges)):
                if i + 1 <= len(candle_ranges) - 1:
                    if candle_ranges[i] < timestamp and candle_ranges[i+1] > timestamp:
                        trades_per_period_length[i].append(line)
                        byte_locations[i].append(byte_copy)
                        candle_start = i
                        break
            # Increments the while loopd
            byte_copy = f.tell()
        return trades_per_period_length, byte_locations

    def get_candles_without_get_request(self, symbol, file_name, start_time_iso, end_time_iso, period_len_minutes):
        """Function to get candles of specified period_length from the start time to the end time

        Args:
            symbol (string): symbol that will appear on the trades
            file_name (string): file where the trades appear
            start_time_iso (string): start time to collect candles iso format
            end_time_iso (string): end time to collect candles iso format
            period_len_minutes (int): lengh that each candle will summarize

        Raises:
            ValueError: if the symbol does not appear in the filename

        Returns:
            list of Candle object: candle object stores open low high close in a organized way
        """
        periods_allowed = [1,3,5,15,30,60,120,240,360,480,720,1440,4320,10080,43200]
        periods_reverse = periods_allowed[::-1]
        left_to_go = []
        for idx, period in enumerate(periods_reverse):
            if period == period_len_minutes:
                left_to_go = periods_reverse[idx:]
        
        if not isinstance(start_time_iso, str):
            start_time_iso = datetime.isoformat(start_time_iso)
        if not isinstance(end_time_iso, str):
            end_time_iso = datetime.isoformat(end_time_iso)
        if symbol not in file_name:
            raise ValueError("symbol string not in filename")
        start_obj = datetime.fromisoformat(start_time_iso)
        
        end_obj = datetime.fromisoformat(end_time_iso)
        print('start_obj: ', start_obj, 'end_obj: ', end_obj)
        # The helper_binary_find_target_time gets the target time in the file using binary search, the boolean declares to get the trade After the target or before the target because some trades don't happen in specific periods 
        line_bytes_start = self.helper_binary_find_target_time_iso(file_name, start_obj, False)
        line_bytes_end = self.helper_binary_find_target_time_iso(file_name, end_obj, True)
        # The get_range_for_candles returns a list of datetime objects with normalized ranges
        candles_dict = {}
        bytes_dict = {}
        for period in left_to_go:
            candle_ranges = self.get_range_for_candles(start_time_iso=line_bytes_start[0][2], end_time_iso=line_bytes_end[0][2], period_len_minutes=period)
            # The helper_loop_through_file goes through the bytes of the file that are relevant
            trades_per_period_length, bytes_locations = self.helper_loop_through_file(file_name=file_name, byte_start=line_bytes_start[1], byte_end=line_bytes_end[1], candle_ranges=candle_ranges)    
            all_candles = self.helper_create_candles(symbol, trades_per_period_length, candle_ranges)
            candles_dict[period] = all_candles
            bytes_dict[period] = bytes_locations
        return candles_dict, bytes_dict


    # def get_candles_on_timestamps(self, symbol, start_date_iso, end_date_iso, period_len_minutes):
        # self.server_obj.get_last_candles(symbol=symbol, start_date_iso, end_date_iso)
    
    # def get_candles_on_timestamps(self, symbol, start_date_iso, end_date_iso, period_len_minutes):
    #     """Function to get candles and make sure they fall into the start date and end date 

    #     Args:
    #         symbol (string): symbol appearing in the candles that are obtained
    #         start_date_iso (string): start date to get candles from
    #         end_date_iso (string): end date to get candles from
    #         period_len_minutes (int): number of minutes for the period length

    #     Returns:
    #         Object Candle: Candle objects to organize open low high close
    #     """
    #     if not isinstance(start_date_iso, str):
    #         start_date_iso = datetime.isoformat(start_date_iso)
    #     if not isinstance(end_date_iso, str):
    #         end_date_iso = datetime.isoformat(end_date_iso)

    #     candles_dict, bytes_dict = self.get_candles_without_get_request(symbol=symbol, file_name=symbol + '.csv', start_time_iso=start_date_iso, end_time_iso=end_date_iso, period_len_minutes=period_len_minutes)
    #     return_candles = {}
    #     return_bytes_locations = {}
    #     # Function get_candles_without_get_request is not entirely accurate so this for loop assures specified date range
    #     all_keys_candles = candles_dict.keys()
    #     for key in all_keys_candles:
    #         gotten_candles_at_key = candles_dict[key]
    #         bytes_locations = bytes_dict[key]
    #         return_candles[key] = []
    #         return_bytes_locations[key] = []
    #         for idx_candle, candle in enumerate(gotten_candles_at_key):
    #             if candle.timestamp >= datetime.fromisoformat(start_date_iso) and candle.timestamp <= datetime.fromisoformat(end_date_iso):
    #                 return_candles[key].append(candle)
    #                 return_bytes_locations[key].append(bytes_locations[idx_candle])
    #     return return_candles, return_bytes_locations
