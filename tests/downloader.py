import copy
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta
import pytz

from exchanges import Exchange, binanceus
from exchanges.binance import binance
from exchanges.Exchange import Candle, Exchange


class downloader:
    def __init__(self, start_time, end_time, symbol):
        self.cusion_seconds = 3600
        self.handle_start(start_time, end_time)
        self.exchange = binanceus
        self.indicies = None
        self.symbol = symbol
        self.file = None
        self.increment = None
        self.used_start = None
        self.counter_use_function = 0
        if not isinstance(self.exchange, Exchange):
            self.exchange = self.exchange()

    def handle_start(self, start_time, end_time):
        initial_time = datetime.fromisoformat(start_time)
        end_time = datetime.fromisoformat(end_time)
        delta = timedelta(seconds=self.cusion_seconds)
        self.start_time = datetime.isoformat(initial_time - delta)
        self.end_time = datetime.isoformat(end_time - delta)

    def get_indicies(self, furthest_back_time_seconds, manager=None):
        date_obj_start = datetime.fromisoformat(self.start_time)
        self.used_start = (date_obj_start - timedelta(seconds=furthest_back_time_seconds)).isoformat()
        self.file = self.symbol + '.csv'
        self.exchange.download_symbol(self.used_start, self.end_time, self.symbol)
        self.used_start = datetime.isoformat(datetime.fromisoformat(self.used_start) + timedelta(seconds=self.cusion_seconds))
        return True, self.used_start

    def get_candles_without_get_request(self, symbol, file_name, start_time_iso, end_time_iso, period_len_minutes, periods_to_use=None):
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
        if periods_allowed is not None:
            left_to_go = periods_to_use
        else:
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
        line_bytes_start = self.helper_binary_find_target_time_iso(file_name, start_obj, after=False)
        line_bytes_end = self.helper_binary_find_target_time_iso(file_name, end_obj, after=False, last=True)
        if line_bytes_end == -1 or line_bytes_end == -1:
            raise ValueError('These should have been populated in the get_indicies function before it')
        # The get_range_for_candles returns a list of datetime objects with normalized ranges
        candles_dict = {}
        bytes_dict = {}
        for period in left_to_go:
            candle_ranges = self.get_range_for_candles(start_time_iso=line_bytes_start[0][2], end_time_iso=line_bytes_end[0][2], period_len_minutes=period)
            # The helper_loop_through_file goes through the bytes of the file that are relevant
            # very very expensive function takes a lot of time
            trades_per_period_length, bytes_locations = self.helper_loop_through_file(file_name=file_name, byte_start=line_bytes_start[1], byte_end=line_bytes_end[1], candle_ranges=candle_ranges)    
            all_candles = self.helper_create_candles(symbol, trades_per_period_length, candle_ranges)
            candles_dict[period] = all_candles
            bytes_dict[period] = bytes_locations
        return candles_dict, bytes_dict

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
        while byte_copy < byte_end:
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

    def helper_binary_find_target_time_iso(self, file, target_date_obj, after=None, last = False):
        """Function to binary search through the file

        Args:
            file (string): file to search through
            target (string): time_iso_format to look for in the file
            after (bool, optional): Since the exact time may not exist it gets the trade that happened after the target time(True) or the first trade before it(False). Defaults to None.
        """
        print(target_date_obj)
        f = open(file, 'r')
        file_size = os.fstat(f.fileno()).st_size
        left = 0
        right = file_size
        compare_me = None
        target = target_date_obj
        # NOTE: ONLY WORKS IF THERE ISN'T A SPACE AT THE END OF THE FILE
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
            try:
                compare_me = datetime.fromisoformat(line[2])
            except:
                if last:
                    return [line1, capture_f_position]
                else:
                    raise ValueError('something went wrong')
                    
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
        if last:
            # print('line', line)
            # print('line1', line1)
            # print('compare_me', compare_me)
            # print('compare_me1', compare_me1)
            return [line1, capture_f_position]
        return -1

    def get_candles_on_timestamps(self, symbol, start_date_iso, end_date_iso, period_len_minutes, periods_allowed=None):
        """Function to get candles and make sure they fall into the start date and end date 

        Args:
            symbol (string): symbol appearing in the candles that are obtained
            start_date_iso (string): start date to get candles from
            end_date_iso (string): end date to get candles from
            period_len_minutes (int): number of minutes for the period length

        Returns:
            Object Candle: Candle objects to organize open low high close
        """
        if not isinstance(start_date_iso, str):
            start_date_iso = datetime.isoformat(start_date_iso)
        if not isinstance(end_date_iso, str):
            end_date_iso = datetime.isoformat(end_date_iso)
        if periods_allowed is not None:
            candles_dict, bytes_dict = self.get_candles_without_get_request(symbol=symbol, file_name=symbol + '.csv', start_time_iso=start_date_iso, end_time_iso=end_date_iso, period_len_minutes=period_len_minutes, periods_to_use=periods_allowed)
        else:
            candles_dict, bytes_dict = self.get_candles_without_get_request(symbol=symbol, file_name=symbol + '.csv', start_time_iso=start_date_iso, end_time_iso=end_date_iso, period_len_minutes=period_len_minutes, periods_to_use=periods_allowed)
        return_candles = {}
        return_bytes_locations = {}
        # Function get_candles_without_get_request is not entirely accurate so this for loop assures specified date range
        all_keys_candles = list(candles_dict.keys())
        for key in all_keys_candles:
            gotten_candles_at_key = candles_dict[key]
            bytes_locations = bytes_dict[key]
            return_candles[key] = []
            return_bytes_locations[key] = []
            for idx_candle, candle in enumerate(gotten_candles_at_key):
                if candle.timestamp >= datetime.fromisoformat(start_date_iso) and candle.timestamp <= datetime.fromisoformat(end_date_iso):
                    return_candles[key].append(candle)
                    return_bytes_locations[key].append(bytes_locations[idx_candle])
        return return_candles, return_bytes_locations

    def send_candles_and_bytes_to_file(self, candles_dict, bytes_dict, start_time, end_time, period_len):
        my_df = pd.DataFrame()
        all_sizes = list(candles_dict.keys())
        all_dicts = []
        for size in all_sizes:
            current_list = candles_dict[size]
            for idx, candle in enumerate(current_list):
                candle_dict = candle.to_dict()
                candle_bytes = bytes_dict[size][idx]
                candle_dict['size'] = size
                candle_dict['bytes'] = candle_bytes
                all_dicts.append(candle_dict)
        my_df = my_df.from_records(all_dicts)
        my_start = start_time.replace(':','_')
        my_end = end_time.replace(':','_')
        my_df.to_csv(f'{my_start}_{my_end}_{period_len}.csv')

    def helper_get_correct_periods(self, increment):
        final_periods_to_use = []
        periods_to_use = []
        periods = [1, 3, 5, 15, 30, 60, 120, 240, 360, 480, 720, 1440, 4320, 10080, 43200]
        temp_use = copy.copy(increment)
        for element in periods:
            if element <= increment:
                periods_to_use.append(element)
        reverse_use = periods_to_use[::-1]
        for idx, element in enumerate(reverse_use):
            if temp_use % element == 0:
                final_periods_to_use.append(element)
        return final_periods_to_use

    def helper_make_dir_if_needed(self, dir_name):
        if not os.path.exists(dir_name):
        # Create the directory only if it doesn't exist
            os.mkdir(dir_name)
        else:
            print(f"The directory '{dir_name}' already exists.")

    def helper_loop_through_names(self, periods, dir_name, symbol):
        for period in periods:
            if os.path.isfile('./' + dir_name + '/' + symbol + 'candles_' + str(period) + '.csv'):
                continue
            else:
                return False
        return True

    def helper_get_from_file(self, periods, dir_name, symbol):
        all_dfs_dict = {}
        for period in periods:
            filename = './' + dir_name + '/' + symbol + 'candles_' + str(period) + '.csv' 
            my_df = pd.read_csv(filename)
            print('df', my_df)
            all_dfs_dict[period] = my_df
        return all_dfs_dict

    def get_dfs_from_file(self, increment, used_start):
        self.increment=increment
        self.counter_use_function+=1
        if self.counter_use_function > 2:
            raise ValueError('trying to prevent infinite case downloader')
        all_dfs_dict = {}
        my_start = self.start_time.replace(':','_')
        my_end = self.end_time.replace(':','_')
        self.final_periods_to_use = self.helper_get_correct_periods(increment)
        dir_name = f'{my_start}_{my_end}'
        self.helper_make_dir_if_needed(dir_name)
        files_exist = self.helper_loop_through_names(periods=self.final_periods_to_use, dir_name=dir_name, symbol=self.symbol)
        if files_exist:
            all_dfs_dict = self.helper_get_from_file(periods=self.final_periods_to_use, dir_name=dir_name, symbol=self.symbol)
        else:
            candels_dict, bytes_dict = self.get_candles_on_timestamps(symbol=self.symbol, start_date_iso=used_start, end_date_iso=self.end_time, period_len_minutes=self.increment, periods_allowed=self.final_periods_to_use)
            all_dfs = self.create_files(bytes_dict=bytes_dict, candles_dict=candels_dict, period_len_minutes=self.increment)
            self.create_csv(candles_df_dict=all_dfs, symbol=self.symbol, dir_name=dir_name)
            self.get_dfs_from_file(increment=self.increment, used_start=self.used_start)
        return all_dfs_dict

    def create_files(self, bytes_dict, candles_dict, period_len_minutes):
        all_dfs = {}
        # del bytes_dict[3]
        # del candles_dict[3]
        candles_keys = list(candles_dict.keys())
        for idx, key in enumerate(candles_keys):
            largest_idx = 0
            candles = candles_dict[key]
            bytes = bytes_dict[key]
            full_aligned = []
            if idx + 1 < len(candles_keys):
                next_candles = candles_dict[candles_keys[idx+1]]
                next_bytes = bytes_dict[candles_keys[idx+1]]   
            if next_candles is not None:
                for candle in candles:
                    candles_next_aligned = []
                    for idx, candle_next in enumerate(next_candles):
                        start_threshold = candle.timestamp    
                        end_threshold = start_threshold + timedelta(seconds=key*60)
                        candle_next_timestamp = candle_next.timestamp
                        if candle_next_timestamp >= start_threshold and candle_next_timestamp <= end_threshold:
                            candles_next_aligned.append(largest_idx)
                            largest_idx += 1
                        if candle_next_timestamp > end_threshold:
                            next_candles =  next_candles[idx:]
                            temp_aligned = copy.copy(candles_next_aligned)
                            full_aligned.append(temp_aligned)
                            break
            if key != 1:
                current_df = pd.DataFrame.from_records(candle.to_dict() for candle in candles)
                copy_aligned = copy.copy(full_aligned)
                while len(copy_aligned) < len(current_df):
                    copy_aligned.append([])
                current_df['step_below'] = copy_aligned
                all_dfs[key] = current_df
            else:
                current_df = pd.DataFrame.from_records(candle.to_dict() for candle in candles)
                copy_aligned = copy.copy(full_aligned)
                while len(bytes) < len(current_df):
                    bytes.append([])
                current_df['bytes'] = bytes
                all_dfs[key] = current_df
        return all_dfs

    def create_csv(self, candles_df_dict, symbol, dir_name):
        candles_keys = list(candles_df_dict.keys())
        candles_list = []
        i = 0
        candles_list = []
        for key in candles_df_dict:
            candles_list.append(candles_df_dict[key])
        for df in candles_list:
            key = candles_keys[i]
            df.to_csv(
                './' + dir_name + '/' + symbol + 'candles_' + str(key) + '.csv', header=True, mode="w", line_terminator="\n"
            )
            i += 1
        return candles_list
# SYMBOL='BTCUSD'
# start_time = '2023-04-01T00:00:00.000000'
# end_time = '2023-05-03T0:0:0.000000'
# hello2 = 'hello'