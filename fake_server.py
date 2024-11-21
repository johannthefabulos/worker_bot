import copy
import os
import time
from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd
import pandas_ta as ta

from exchanges.Exchange import Candle, Order


class trade_line:
    def __init__(self, line):
        self.price = float(line[4])
        self.volume= float(line[6].replace('/n',''))
        self.timestamp = line[2]
        self.symbol = line[3]
        self.amt_full = 0
        expiration_time = ''

    def to_dict(self):
        return {
            'price': self.price,
            'volume': self.volume,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'amt_full': self.amt_full
        }

class candle_line:
    def __init__(self, line):
        self.symbol = line['symbol']
        self.timestamp = line['timestamp']
        self.high = line['high']
        self.low = line['low']
        self.open = line['open']
        self.last =line['close']

class FakeServer():
    def __init__(self, exchange, start_time_iso, end_time_iso, used_account, trigger_event=None, trigger_queue=None, trigger_id =None):
        self.manager = None
        self.exchange = exchange
        self.global_start_iso = start_time_iso
        self.global_end_iso = end_time_iso
        self.used_account = used_account
        self.trigger_event = trigger_event
        self.trigger_queue = trigger_queue
        self.trigger_id = trigger_id
        self.start_time = None
        self.TOLERANCE = .000000001
        self.current_active_orders = []
        self.candles_so_far = None
        self.symbol = None
        self.dict_of_currency = None
        self.dict_of_dfs = None
        self.start_simulation = False
        self.simulation_df = None
        self.current_line = None
        self.current_series = None
        self.current_candle = None
        self.start_of_period_time = None
        self.market = False
        self.manager = None
        self.should_not_proceed = None
        self.ghost_orders = []
        self.all_undone_objects = []
        self.mongo_obj = None
        # TODO: make this reflect the current time in trades self.current_time_iso
        self.current_time_iso = None
        #TODO: for sake of not getting overwhelmed we are going to assume that this fake server always has the whole amount of the data that its simulating over
        # but it will have to be implemented later
        # we will also assume we have a current element of that df that we are working with
        # we will have a series as the current representation of where in time we are for simulating
    def set_sim_dfs(self, start_time_iso, candles_df, dict_of_candles_dfs):
        self.simulation_df = candles_df
        print('candles_df: ', candles_df)
        for idx, row in enumerate(self.simulation_df.iterrows()):
            row_timestamp = datetime.fromisoformat(row[1]['timestamp'])
            start_time_obj = datetime.fromisoformat(start_time_iso) 
            if row_timestamp >= start_time_obj:
                self.current_series = pd.Series(row)
                self.current_series.name = idx
                print(self.current_series.name)
                break
        print('name', self.current_series.name)
        self.symbol = self.current_series[1]['symbol']
        print('sim_df', self.simulation_df)
        print('candles')
        self.current_time_iso = self.current_series[1]['timestamp']
        self.dict_of_dfs = dict_of_candles_dfs
    
    def set_manager(self, manager):
        self.manager = manager

    def set_globals_and_begin(self, symbol, global_start, global_end):
        self.symbol = symbol
        self.global_start_iso = global_start
        self.global_end_iso = global_end
        self.start_time = self.global_start_iso.copy()

    def replace_order(self, order):
        #TODO: I'm pretty sure that we don't need to replace anything because its a list obtained from the strat itself which is using amount_objs shared state
        #but checkout to make sure.
        swap_counter = 0
        flag_swap_occured = False
        active_orders = []
        for trade in self.current_active_orders:
            if trade.needed_id == order.needed_id and trade.amt_id == order.amt_id:
                active_orders.append(order)
                flag_swap_occured = True
                swap_counter += 1
            else:
                active_orders.append(trade)
        if not flag_swap_occured or swap_counter > 1 or swap_counter == 0:
            raise ValueError("There was an error with replacing orders")
        self.current_active_orders = active_orders

    def make_order_active(self, order):
        if not order.active:
            if order.side == 'sell':
                order.last_time_modified = self.current_time_iso
                order.original_price = order.price
                order.original_sell_time = self.current_time_iso
        self.current_active_orders.append(order)
        order.active = True

    def get_current_price(self, symbol):
        # current_series in nonsubscripible because it is none
        line_bytes = self.helper_binary_find_target_time_iso(my_file=symbol+'.csv', target_date_obj=self.current_series[1]['timestamp'], after=True)
        print('line_bytes', line_bytes)
        return line_bytes[0][4]

    def market_order(self, order):
        order.market = True
        order.placed = True
        self.current_active_orders.append(order)

    def set_dict(self, myDict):
        self.dict_of_currency = myDict

    def get_balance(self):
        return self.dict_of_currency
    
    def get_candles_from_lines(self, lines_df):
        final_candles = []
        for line in lines_df.iterrows():
            line = line[1]
            current_candle = Candle(symbol=line['symbol'],timestamp=line['timestamp'],high=line['high'], low=line['low'],first=line['open'], last=line['close'])
            copy_candle = current_candle.copy()
            final_candles.append(copy_candle)
        return final_candles

    def get_supertrend(self, backup):
        candles_so_far_df = self.simulation_df[self.current_series.name-(backup+1):self.current_series.name]
        # TODO: make sure that if we are here and simulating then the candles are always downloaded so we don't have to use get_candles_on_timestamps which is very very expensive
        # and takes a long time
        st = ta.supertrend(candles_so_far_df['high'], candles_so_far_df['low'], candles_so_far_df['close'], backup, 2)
        print('st: ', st)
        print('candles_so_far_df: ', candles_so_far_df)
        # slope = self.calculate_slope(data=st, supertrend_column='SUPERT_10_2.0', start_idx=20, end_idx=len(st)-1)
        difference = st.iloc[-2]['SUPERT_10_2.0']-st.iloc[-1]['SUPERT_10_2.0']
        return difference

    def periods_passed_now_timestamp(self, timestamp_iso, period_len_minutes):
        now = datetime.fromisoformat(self.current_time_iso)
        past = datetime.fromisoformat(timestamp_iso)
        total_seconds = (now - past).total_seconds()
        total_periods_passed = int(total_seconds/(period_len_minutes*60))
        return total_periods_passed

    def get_last_candles(self, symbol, num_prev_candles, len_period_minutes):
        #TODO: make sure this actually gives the last num_prev_candles
        #TODO: make sure this makes sure that the candles given are of symbol and len_period_minutes
        print('sim_df', self.simulation_df)
        last_lines = self.simulation_df.iloc[self.current_series.name-num_prev_candles:self.current_series.name]
        print('last_lines: ', last_lines)
        return_candles = []
        for line in last_lines.iterrows():
            current_candle = Candle(symbol=symbol,timestamp=line[1]['timestamp'],high=line[1]['high'], low=line[1]['low'],first=line[1]['open'], last=line[1]['close'])
            return_candles.append(current_candle)
        return return_candles

    def helper_binary_find_target_time_iso(self, my_file, target_date_obj, after=None):
            """Function to binary search through the file

            Args:
                file (string): file to search through
                target (string): time_iso_format to look for in the file
                after (bool, optional): Since the exact time may not exist it gets the trade that happened after the target time(True) or the first trade before it(False). Defaults to None.
            """
            try:
                f = open(my_file, 'r')
            except:
                return -1
            file_size = os.fstat(f.fileno()).st_size
            left = 0
            right = file_size
            compare_me = None
            if not isinstance(target_date_obj, datetime):
                if '+' not in target_date_obj:
                    target_date_obj += '+00:00'
                    target = datetime.fromisoformat(target_date_obj)
                else:
                    target = datetime.fromisoformat(target_date_obj)
            else:
                string = datetime.isoformat(target_date_obj)
                if '+' not in string:
                    string += '+00:00'
                    target = datetime.fromisoformat(string)
                else:
                    target = datetime.fromisoformat(string)
            check = datetime.isoformat(target)
            # Generic binary search adapted to bytes
            while left < right:
                mid = (left + right) // 2
                f.seek(mid)
                while f.tell() > 0 and f.read(1) != '\n':
                    f.seek(f.tell() - 2, os.SEEK_SET)
                capture_f_position = f.tell()
                line = f.readline().split(',')
                if line[0] == '':
                    capture_f_position = f.tell()
                    line = f.readline().split(',')
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
    
    def process_locations(self, trade_locations):
        trade_locations = trade_locations.replace('[','')
        trade_locations = trade_locations.replace(']','')
        trade_locations = trade_locations.replace(' ','')
        trade_list = trade_locations.split(',')
        return trade_list


    def convert_locations_to_trades(self, symbol, all_trades_locations):
        print('all_trades_locations: ', all_trades_locations)
        all_trades_locations = self.process_locations(all_trades_locations)
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

    def simulator_control(self):
        start_time = time.time()
        for idx, line in enumerate(self.simulation_df.iterrows()):
            if idx < self.current_series.name:
                continue
            self.current_candle = Candle(symbol=self.symbol,timestamp=line[1]['timestamp'],high=line[1]['high'], low=line[1]['low'],first=line[1]['open'], last=line[1]['close'])
            self.handle_trigger_period(self.current_candle)
            self.current_time_iso = line[1]['timestamp']
            self.start_of_period_time = line[1]['timestamp']
            self.current_series = pd.Series(line)
            self.current_series.name = idx
            print(self.current_series.name)
            self.manager.strat.decision()
            self.should_not_proceed = (self.current_candle.timestamp + timedelta(seconds=5*60)).isoformat()
            # Iteresting side effect of placing orders is we have tier last first and tier initial last
            print('active_orders', self.current_active_orders)
            # After decision we should have a populated self.current_active_orders
            # And since thats the case we are going to simulate the period
            keep_going = True
            while keep_going:
                return_sim = self.simulate_period(line, wait_time_for_order=1)
                if return_sim == -1:
                    keep_going = False
            total = 0
            for amt_obj in self.manager.strat.list_of_objs:
                print('id: ', amt_obj.amt_obj_id, ' profit: ', amt_obj.profit)
                total+=amt_obj.profit
            print('total: ', total)
        for amt_obj in self.manager.strat.list_of_objs:
            print('id: ', amt_obj.amt_obj_id, ' profit: ', amt_obj.profit)
            total+=amt_obj.profit
        print('total: ', total)
        end_time = time.time()
        total_time = end_time - start_time
        print(total_time)
        print('finished!')

    def handle_trigger_trade(self, order):
        if self.trigger_event is not None:
            trigger_dict = {}
            trigger_dict['value'] = order.price
            trigger_dict['name']  = self.current_time_iso
            trigger_dict['side'] = order.side
            trigger_dict['trigger_id'] = self.trigger_id
            self.trigger_queue['trade'].put(trigger_dict)
            self.trigger_event['trade'].set()

    def handle_trigger_period(self, candle):
        if self.trigger_event is not None:
            trigger_dict = {}
            trigger_dict['value'] = candle.high
            trigger_dict['name']  = self.current_time_iso
            trigger_dict['trigger_id'] = self.trigger_id
            self.trigger_queue['period'].put(trigger_dict)
            self.trigger_event['period'].set()
        # prices_dict = {}
        # price = self.get_current_price(candle.symbol)
        # prices_dict[candle.symbol] = price
        # self.mongo_obj.append_element_col(name='times_btc', element={
        #     "name": self.current_time_iso[:self.current_time_iso.find('+')], "value": float(candle.high)}
        # )
        # total = self.used_account.find_total_usd(self.current_time_iso, prices_dict)
        # self.mongo_obj.append_element_col(name='funds', element={'funds': str(total)})



    def helper_order_handler(self, order):
        counter = 0
        new_active_orders = []
        for current_active in self.current_active_orders:
            if current_active.amt_id == order.amt_id and current_active.needed_id == order.needed_id:
                counter+=1
            else:
                new_active_orders.append(current_active)
        if counter != 1:
            raise ValueError('Only one order at a time should be sent to active')
        self.handle_trigger_trade(order)
        self.current_active_orders = new_active_orders
        new_order = self.manager.add_to_path(order)
        if new_order.side == 'sell':
            self.manager.strat.order_sell_handler(new_order)
        else:
            self.manager.strat.order_buy_handler(new_order)

    def get_max_price(self, list_to_iterate):
        max_price = max([trade.price for trade in list_to_iterate])
        return max_price
    
    def get_current_time(self):
        return self.current_time_iso

    def simulate_period(self, line, wait_time_for_order):
        if len(self.current_active_orders) == 0:
            return -1
        if not isinstance(self.current_candle.timestamp,str):
            # TODO: make sure this if never runs 
            self.current_candle.timestamp = self.current_candle.timestamp.isoformat()
            # raise ValueError("current_candle must be in str format")
        if self.check_if_skippable(): 
            return -1
        byte_locations = self.get_relevant_data(self.symbol, line, self.dict_of_dfs, self.current_time_iso)
        if byte_locations is None or byte_locations == []:
            return -1
        all_trades = self.convert_locations_to_trades(self.symbol, byte_locations)
        if all_trades == []:
            return -1
        all_trades_df = pd.DataFrame.from_records([trade.to_dict() for trade in all_trades])
        print('all_trades_df', all_trades_df)
        reset = False
        # TODO: last left off check_for expiration giving none type prices and market orders should always be filled
        # used to be deepcopy
        iterate_over_list = copy.copy(self.current_active_orders)
        all_resets = []
        all_resets.append(False)
        for idx, trade_obj in enumerate(all_trades_df.iterrows()):
            print('trade_obj', trade_obj)
            if True in all_resets:
                if not isinstance(self.current_candle.timestamp, str):
                    self.current_candle.timestamp = datetime.isoformat(self.current_candle.timestamp)
                # used to be deepcopy lets hope it works
                iterate_over_list = copy.copy(self.current_active_orders)
                reset = False
                all_resets = []
            for order in iterate_over_list:
                print(order.price)
                print(Decimal(trade_obj[1]['price']))
                print(order.price >= Decimal(trade_obj[1]['price']))
                amt_full_var = trade_obj[1]['amt_full']
                if amt_full_var == 'full':
                    allowed_size_order = 0
                else:
                    allowed_size_order = abs(Decimal(trade_obj[1]['amt_full']) - Decimal(trade_obj[1]['volume']))
                if (order.side == 'sell' and order.price <= Decimal(trade_obj[1]['price'])) or (order.side == 'buy' and order.price >= Decimal(trade_obj[1]['price']) or order.market):
                    reset, skip_order = self.check_for_expiration(self.all_undone_objects, trade_obj, order)
                    all_resets.append(reset)
                    if skip_order:
                        break
                    if allowed_size_order < order.total_amt and allowed_size_order != 0:
                        order.filled += Decimal(allowed_size_order)
                        trade_obj[1]['amt_full'] = 'full'
                        order.expired = datetime.fromisoformat(trade_obj[1]['timestamp']) + timedelta(seconds = wait_time_for_order * 60)
                        order.traded_price = trade_obj[1]['price']
                        order.timestamp = trade_obj[1]['timestamp']
                        self.all_undone_objects.append(order)
                        all_resets.append(False)
                    elif allowed_size_order > order.total_amt and allowed_size_order !=0: 
                        order.timestamp = trade_obj[1]['timestamp']
                        trade_obj[1]['amt_full'] = str(Decimal(trade_obj[1]['amt_full']) + order.total_amt)
                        if abs(Decimal(trade_obj[1]['amt_full']) - Decimal(trade_obj[1]['volume'])) < self.TOLERANCE:
                            trade_obj[1]['amt_full'] = 'full'
                        order.traded_price = trade_obj[1]['price']
                        order.price = order.traded_price
                        self.helper_order_handler(order)
                        self.used_account.record(order)
                        all_resets.append(True)
                    else:
                        all_resets.append(True)
            self.current_time_iso = trade_obj[1]['timestamp']
        return 1

    def correct_string(self, processed_string):
        processed_string = processed_string.replace('[', '')
        processed_string = processed_string.replace(']', '')
        processed_string = processed_string.replace(' ', '')
        processed_list = processed_string.split(',')
        processed_list = [int(single_element) for single_element in processed_list]
        return processed_list

    def get_relevant_data(self, symbol, candle, candles_dict, start_time_obj):
            # Deals with step below things which means there is a downloaded file which each increment from self.params['lookback'] down to one where one will have the 
            # trades so there is not efficiency wasted 
            # I'm pretty confused with what I was trying to accomplish with determine_valid_candle.....why not just check if each candle contains the order price??
            candles_keys = list(candles_dict.keys())
            i = 0
            temp_indicies = [[self.current_series.name]]
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
                    # current_candle = Candle(symbol=symbol,timestamp=line[1]['timestamp'],high=line[1]['high'], low=line[1]['low'],first=line[1]['open'], last=line[1]['close'])
                    current_candle = Candle(symbol=symbol,timestamp=line['timestamp'],high=line['high'], low=line['low'],first=line['open'], last=line['close'])
                    if current_key != 1:
                        valid_candle = self.determine_valid_candle(current_candle, self.current_active_orders, start_time_obj=start_time_obj, not_at_one=True)
                        if valid_candle == True:
                            num_valid_candle+=1
                            byte_locations_string = current_candle_df.iloc[name]['step_below']
                            correct_list = self.correct_string(byte_locations_string)
                            temp_indicies.append(correct_list)
                            break
                        elif valid_candle == -1:
                            byte_locations_string = current_candle_df.iloc[name]['step_below']
                            correct_list = self.correct_string(byte_locations_string)
                            temp_indicies.append(correct_list)
                            break
                        elif valid_candle == False:
                            continue
                        # current_candle_df.iloc[name]['step_below']
                    else:
                        valid_candle = self.determine_valid_candle(current_candle, self.current_active_orders, start_time_obj=start_time_obj)
                        if valid_candle == True:
                            print('candle_df: ', current_candle_df)
                            return current_candle_df.iloc[name]['bytes']
                        elif valid_candle == -1:
                            continue
            return None

    def check_for_expiration(self, all_undone_objects, trade_obj, order):
        reset = False
        elements_to_break = []
        skip_order = False
        for idx, element in enumerate(all_undone_objects):
            trade_obj_time = datetime.fromisoformat(trade_obj[1]['timestamp'])
            if trade_obj_time >= element.expired:
                element.total_amt = element.filled
                order.price = order.traded_price
                self.helper_order_handler(order)
                self.used_account.record(order)
                elements_to_break.append(idx)
                reset = True
                skip_order = True
            elif element.needed_id == order.needed_id:
                if element.filled + Decimal(trade_obj[1]['volume']) >= element.total_amt:
                    element.timestamp = trade_obj[1]['timestamp']
                    order.traded_price = element.traded_price
                    order.timestamp = trade_obj[1]['timestamp']
                    order.price = order.traded_price
                    self.helper_order_handler(order)
                    self.used_account.record(order)
                    elements_to_break.append(idx)
                    skip_order = True
                    reset = True
                elif element.filled + Decimal(trade_obj[1]['volume']) < element.total_amt:
                    element.filled += Decimal(trade_obj[1]['volume'])
                    element.timestamp = trade_obj[1]['timestamp']
                    skip_order = True
                    reset = True
        for element in elements_to_break:
            try:
                all_undone_objects.pop(element)
            except:
                pass
        return reset, skip_order

    def determine_valid_candle(self, current_candle, all_trades, start_time_obj=None, not_at_one = False):
        if start_time_obj:
            time_object = datetime.fromisoformat(start_time_obj)
            if not isinstance(current_candle.timestamp, str):
                current_candle.timestamp = current_candle.timestamp.isoformat()
            candle_obj = datetime.fromisoformat(current_candle.timestamp)
            print(time_object)
            print(candle_obj)
        num_failed = 0
        for current_order in all_trades:
            if current_order.market:
                return True
            if ((current_order.side == 'sell' and current_order.price > current_candle.high) or (current_order.side == 'buy' and current_order.price < current_candle.low)):
                num_failed+=1
        if num_failed == len(all_trades):
            return False
        elif candle_obj == time_object:
            return -1
        elif start_time_obj and candle_obj > time_object:
            return True
        elif not_at_one:
            return -1
        elif start_time_obj:
            return False
        elif not start_time_obj:
            return True
        return False

    def check_if_skippable(self):
        skip = False
        number_found = 0
        for order in self.current_active_orders:
            if order.market:
                return False
            if (order.side == 'sell' and order.price > self.current_candle.high) or (order.side == 'buy' and order.price < self.current_candle.low):
                skip = True
                number_found+=1
            else:
                point_here = 1
        if skip and number_found == len(self.current_active_orders):
            return True
        else:
            return False

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
