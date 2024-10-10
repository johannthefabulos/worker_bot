import copy
import datetime
import hashlib
import hmac
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import *

import dateutil
import pandas as pd
import pytz
import requests

from exchanges.Exchange import Candle, Exchange, Order, Symbol, Trade
from exchanges.socket import *
from manage.managetrades import suggested_trade


class binance(Exchange):
    def __init__(
        self,
        public_key=None,
        secret=None,
        auth=True,
        trigger_dict=None,
        queue_dict=None,
        endpoint="https://api.binance.com",
        ws_endpoint="wss://stream.binance.com:9443",
        name_override="BINANCE",
    ):
        super().__init__(
            name_override,
            endpoint,
            auth=auth,
            rest_requests=20,
            socket_requests=1000000,
        )
        self.trigger_dict = trigger_dict
        self.queue_dict = queue_dict
        print('queue_dict', queue_dict, 'trigger_dict', trigger_dict)
        self.public_key = public_key
        self.secret = secret
        self.ws_endpoint = ws_endpoint
        self.keep_alive_thread = None
        self.last_socket_update = 0
        self.lock = threading.Lock()
        self.maker_fee = 0.00075
        self.taker_fee = 0.00075
        self.orders = []
        self.wait_time_minutes = 1
        self.all_undone_objects = []
        self.TOLERANCE = 0.000000001
        self.needs_pop = False
        self.retire = []
        self.ws_streams = (
            {}
        )  # Dictionary of stream info ex: {'account':{<listen_key>:<ws_object>}, 'orderbook':{<symbol>:<ws_object>}}

        self.refresh()
        self.add_callbacks(on_order_executed=self.on_trade_fulfilled, on_market_trade=self.on_market_fulfilled)

    def on_market_fulfilled(self, order: Order, trade: Trade):
        print('VOLUME', trade.volume)
        current_saved = None
        counter = 0
        saved_order = None
        current_idx = None
        for idx, current_saved in enumerate(self.orders):
            if order.order_id == current_saved.assigned_id:
                counter+=1
                saved_order = current_saved
                current_idx = idx
        if saved_order is None or counter != 1:
            raise ValueError('it should have been found')
        with self.lock:
            saved_order.filled+=trade.volume
        filled_order = suggested_trade(saved_order.needed_id, trade.timestamp, saved_order.filled, saved_order.side, saved_order.parent_id, saved_order.manager_id, saved_order.parent_price, saved_order.symbol, saved_order.additional_info)
        filled_order.amt_id = saved_order.amt_id
        filled_order.market=True
        filled_order.price = trade.price
        with self.lock:
            print('completed', order.completed, ' ', saved_order.filled)
            if order.completed:
                self.orders.pop(current_idx)
                self.needs_pop = False
                self.handle_order(saved_order, filled_order, current_idx, needs_pop=self.needs_pop)

    def on_trade_fulfilled(self, order: Order, trade: Trade):
        with self.lock:
            counter = 0
            current_idx = None
            for idx, current_saved in enumerate(self.orders):
                if order.order_id == current_saved.assigned_id:
                    counter += 1
                    saved_order = current_saved
                    current_idx = idx
            if counter != 1:
                raise ValueError('it should have been found')
            proceed = self.check_undone_objects(saved_order, trade.volume, current_idx, trade.timestamp, order.completed)
            saved_order.timestamp = trade.timestamp
            if proceed and order.completed:
                # if an order has expired it means the time to wait is over...not that it is not longer a viable trade so we only have the amount be the difference
                filled_order = suggested_trade(saved_order.needed_id, trade.timestamp, saved_order.total_amt - saved_order.filled, saved_order.side, saved_order.parent_id, saved_order.manager_id, saved_order.parent_price, saved_order.symbol, saved_order.additional_info)
                filled_order.completed = order.completed
                filled_order.price = saved_order.price
                filled_order.amt_id = saved_order.amt_id
                self.handle_order(saved_order, filled_order, current_idx, trade.timestamp)
            elif proceed and not order.completed:
                saved_order.expired = datetime.utcnow() + timedelta(seconds=self.wait_time_minutes*60)
                saved_order.filled += trade.volume
                saved_order.timestamp = trade.timestamp
                self.all_undone_objects.append(saved_order)

    def handle_order(self, saved_order, filled_order, idx, needs_pop=True):
        #last change changed saved order to filled order in next two lines untested don't know if it will work
        new_order = filled_order.manager_id.add_to_path(filled_order)
        current_manager = filled_order.manager_id
        if idx is not None:
            if needs_pop:
                self.orders.pop(idx)
        self.handle_trigger_trade(filled_order)
        if new_order.side == 'sell':
            current_manager.strat.order_sell_handler(new_order)
        else:
            current_manager.strat.order_buy_handler(new_order)

    def release_undone(self):
        release = []
        for element in self.retire:
            needed_id = element[0]
            amt_id = element[1]
            for idx,undone_obj in enumerate(self.all_undone_objects):
                if undone_obj.needed_id == needed_id and undone_obj.amt_id == amt_id:
                    release.append(idx)
        for realease_me in release:
            self.all_undone_objects.pop(realease_me)

    def check_undone_objects(self, order, volume, idx, timestamp, completed):
        self.release_undone()
        for undone_obj in self.all_undone_objects:
            if undone_obj.expired < datetime.utcnow():
                filled_order = suggested_trade(undone_obj.needed_id, str(datetime.utcnow()), undone_obj.filled, undone_obj.side, undone_obj.parent_id, undone_obj.manager_id, undone_obj.parent_price, undone_obj.symbol, undone_obj.additional_info)
                filled_order.amt_id = undone_obj.amt_id
                filled_order.price = undone_obj.price
                filled_order.completed = undone_obj.completed
                filled_order.tier_assigned = undone_obj.tier_assigned
                # self.retire.append([order.needed_id, order.amt_id])
                self.handle_order(order, filled_order, idx)
                return False
            if order is None:
                return
            if undone_obj.needed_id == order.needed_id and order.amt_id == undone_obj.amt_id:
                if timestamp is not None:
                    timestamp_to_use = timestamp
                else:
                    timestamp_to_use = order.timestamp
                if completed:
                    filled_order = suggested_trade(order.needed_id, timestamp_to_use, order.total_amt, order.side, order.parent_id, order.manager_id, order.parent_price, order.symbol, order.additional_info)
                    filled_order.amt_id = order.amt_id
                    filled_order.completed = True
                    filled_order.price = order.price
                    filled_order.tier_assigned = undone_obj.tier_assigned
                    self.retire.append([order.needed_id, order.amt_id])
                    self.handle_order(order, filled_order, idx)
                    return False
                elif ((Decimal(order.filled) + Decimal(volume)) > Decimal(order.total_amt)):
                    raise ValueError('something went wrong in exchange')
                elif not completed:
                    order.filled += volume
                    return False
        return True

    def handle_trigger_trade(self, order):
        if self.trigger_dict is not None:
            print('order', order)
            temp = {}
            temp['value'] = order.price
            # seconds = order.timestamp / 1000
            # iso_time = datetime.utcfromtimestamp(seconds).isoformat()
            # print('live_trade: ', iso_time)
            temp['name'] = order.timestamp
            print('order_timestamp', order.timestamp)
            temp['side'] = order.side
            temp['trigger_id'] = 1
            self.queue_dict['trade'].put(temp)
            self.trigger_dict['trade'].set()

    def handle_pending(self, pending_orders):
        if self.trigger_dict is not None:
            temp = []
            for value in pending_orders:
                current_dict = {}
                current_dict['total_amt'] = float(value.total_amt)
                current_dict['value'] = float(value.price)
                current_dict['name'] = datetime.utcnow().isoformat()
                current_dict['side'] = value.side
                temp.append(current_dict)

            self.queue_dict['pending'].put(temp)
            self.trigger_dict['pending'].set()    

    def date_to_timestamp(self, timestamp):
        if type(timestamp) is str:
            return dateutil.parser.parse(timestamp)
        else:
            return timestamp

    def date_to_milliseconds(self, date_str):
        """Convert UTC date to milliseconds
        If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
        See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/
        :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
        :type date_str: str
        """
        # get epoch value in UTC
        epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
        # parse our date string
        d = self.date_to_timestamp(date_str)
        # if the date is not timezone aware apply UTC timezone
        if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
            d = d.replace(tzinfo=pytz.utc)

        # return the difference in time
        return int((d - epoch).total_seconds() * 1000.0)

    def authenticate(self, data):
        """Authenticate data with secret key"""
        hash = hmac.new(self.secret.encode(), data.encode(), hashlib.sha256)
        return str(hash.hexdigest())

    def get_rest(self, request, request_type="get", data={}, requires_auth=False):
        data_dict = data.copy()

        if requires_auth:
            data_dict["timestamp"] = int(time.time() * 1000)
        data_str = ""
        values = request.split("?")
        if len(values) == 2:
            data_str = values[1]
        if data_dict:
            for key, value in data_dict.items():
                if len(data_str) > 0:
                    data_str += "&"

                data_str += str(key) + "=" + str(value)
        if requires_auth:
            request += "?"
            if len(data_str) > 0:
                request += data_str + "&"
            request += "signature=" + self.authenticate(data_str)
        elif len(data_str) > 0:
            request += "?" + data_str

        self.add_msg("REQUEST: " + str(request))
        #print("REQUEST:", request)
        response = super().get_rest(request, request_type=request_type)
        if "code" in response:
            self.add_msg("Error: " + str(response["msg"]))
            code = int(response["code"])
            if code == 1003 or code == 1015:
                wait_time = 10000
                matches = re.search(r"until ([0-9]*\.", response["msg"])
                if matches and len(matches.groups()) > 0:
                    wait_time = int(matches.groups()[0]) - int(time.time() * 1000)
            time.sleep(wait_time / 1000)
        return response

    def convert_time(self, time):
        return (datetime.utcfromtimestamp(float(time) / 1000.0).replace(
            tzinfo=timezone.utc
        ))

    def get_orderbook(self, symbol, limit=100):
        """Get orderbook."""
        data = {"limit": limit, "symbol": symbol.name}
        book = self.get_rest("/api/v1/depth", data=data)

        buy_orders = []
        for buy_order in book["bids"]:
            buy_orders.append(
                Order(symbol, price=float(buy_order[0]), volume=float(buy_order[1]))
            )
        sell_orders = []
        for sell_order in book["asks"]:
            sell_orders.append(
                Order(symbol, price=float(sell_order[0]), volume=float(sell_order[1]))
            )

        return Orderbook(symbol, buy_orders, sell_orders)

    def get_balance(self, currencies=None, all_funds=False):
        returned_balances = []
        response = self.get_rest("/api/v3/account", requires_auth=True)

        tracked_currencies = {}
        for currency in response["balances"]:
            if (
                currencies is None
                and (float(currency["free"]) > 0 or float(currency["locked"]) > 0)
            ) or (currencies is not None and currency["asset"] in currencies):
                total_balance = float(currency["free"]) + float(currency["locked"])
                tracked_currencies[currency["asset"]] = {
                    "available": float(currency["free"]),
                    "reserved": float(currency["locked"]),
                    "total": total_balance,
                }
        return tracked_currencies

    def get_active_orders(self, symbols=None):
        """Return all orders active for this account"""
        orders = []
        order_data = self.get_rest("/api/v3/openOrders", requires_auth=True)
        for order in order_data:
            symbol = self.get_symbol_from_name(order["symbol"])
            volume = float(order["origQty"]) - float(order["executedQty"])
            orders.append(
                Order(
                    symbol,
                    timestamp=order["time"],
                    order_id=order["clientOrderId"],
                    price=order["price"],
                    volume=volume,
                    side=order["side"].lower(),
                    executed_volume=float(order["executedQty"]),
                )
            )
        return orders

    def new_order(self, order, force_limit=False):

        """Place an order."""
        if not order.active:
            if order.side == 'sell':
                order.last_time_modified = datetime.utcnow().isoformat()
                order.original_price = order.price
                order.active = True
        
        price, volume = self.prepare_order(order)
        assigned_id = str(uuid.uuid4())
        assigned_id = assigned_id.replace('-','')
        order.assigned_id = assigned_id
        if order.symbol_obj.prices_equal(price, 0):
            self.add_msg("Attempt to place order at price of 0")
            return None
        elif order.symbol_obj.volumes_equal(volume, 0):
            self.add_msg("Error: Attempt to place order at volume of 0")
            return None

        if not order.symbol_obj.verify(price, volume):
            self.add_msg(
                "Error: attempt to place order below min notional value (price was "
                + price
                + " and volume was "
                + volume
                + ")"
            )
            return None

        self.add_msg(
            "New "
            + order.side
            + " order for "
            + order.symbol_obj.name
            + " at price of "
            + price
            + " and volume of "
            + volume
        )
        print(
            "New",
            order.side,
            "order for",
            order.symbol_obj.name,
            "at price of",
            price,
            "and volume of",
            volume,
        )
        data = {
            "newClientOrderId": assigned_id,
            "symbol": order.symbol,
            "side": order.side.upper(),
            "quantity": volume,
            "type": "LIMIT",
            "price": price,
            "timeInForce": "GTC",
        }
        response = self.get_rest(
            "/api/v3/order", data=data, request_type="post", requires_auth=True
        )
        if not response:
            self.add_msg("Error: null response")
            return None
        if "error" in response:
            self.add_msg(
                "Order place error: "
                + str(response["error"])
                + " request: "
                + str(data)
            )
            return None
        order.active = True
        self.orders.append(order)
        self.handle_pending(self.orders)
        return response["clientOrderId"]
    
    def round_volume(self, volume, round_up=False):
        """Round volume up or down to nearest increment"""
        rounded = int(float(volume) / 0.00001 + 0.00000000001) * 0.00001
        return rounded + 0.00001 if round_up else rounded

    def market_order(self, order):
        # self.orders.append(order)
        order.market=True
        assigned_id = str(uuid.uuid4())
        assigned_id = assigned_id.replace('-','')
        order.assigned_id = assigned_id
        # self.all_undone_objects.append(order)
        """Place a market order"""
        order.total_amt = round(self.round_volume(order.total_amt), 8)
        self.orders.append(order) 
        print('quantity', order.total_amt)
        print('id', order.assigned_id)
        # filled_order = suggested_trade(order.needed_id, datetime.utcnow(), order.total_amt , order.side, order.parent_id, order.manager_id, order.parent_price, order.symbol, order.additional_info)
        # filled_order.amt_id = order.amt_id
        # order.manager_id.add_to_path(filled_order)
        # volume = order.symbol.string_volume(
        #     order.symbol.round_volume(order.volume, round_up=False)
        # )
        # if order.symbol.volumes_equal(volume, 0):
        #     self.add_msg("Error: attempt to place order at volume of 0", level)
        #     return None

        print(
            "New",
            order.side,
            " market order for",
            order.symbol,
            # order.symbol.name,
            "at volume of",

            order.total_amt,
        )
               # data = {
        #     "symbol": order.symbol.name,
        #     "side": order.side.upper(),
        #     "quantity": volume,
        #     "type": "MARKET",
        # }
        data = {
            "newClientOrderId": assigned_id,
            "symbol": order.symbol,
            "side": order.side.upper(),
            "quantity": order.total_amt,
            "type": "MARKET",
        }
        # if order.order_id:
        #     data["newClientOrderId"] = order.order_id
        response = self.get_rest(
            "/api/v3/order", data=data, request_type="post", requires_auth=True
        )
        if "error" in response:
            self.add_msg(
                "Order place error: "
                + str(response["error"])
                + " request: "
                + str(data)
            )
            return None
        return response["clientOrderId"]

    def get_order(self, order, wait=None):
        """Get order info."""
        symbol = order.symbol
        if not isinstance(symbol, Symbol):
            base = symbol[:symbol.index('USD')]
            quote = symbol[symbol.index('USD')]
            increment = self.get_increment(base)
            symbol_obj = Symbol(symbol, base, quote, base_increment=float(increment))
            order.symbol_obj = symbol_obj
        data = {"symbol": order.symbol, "origClientOrderId": order.needed_id}
        order_dict = self.get_rest("/api/v3/order", data=data, requires_auth=True)
        return Order(symbol=order_dict['symbol'], timestamp=order_dict['time'], order_id=order_dict['clientOrderId'], volume=order_dict['origQty'], side=order_dict['side'], executed_volume=order_dict['executedQty'])

    def cancel_order(self, order):
        my_counter = 0
        for idx,saved_order in enumerate(self.orders):
            if order.assigned_id == saved_order.assigned_id:
                self.orders.pop(idx)
                my_counter+=1
                break
        if my_counter != 1:
            raise ValueError('order not found when')

        """Cancel order."""
        self.add_msg("Cancel order " + str(order))

        data = {"symbol": order.symbol, "origClientOrderId": order.assigned_id}
        self.get_rest(
            "/api/v3/order", data=data, request_type="delete", requires_auth=True
        )

    def cancel_orders(self, buy_orders=True, sell_orders=True):
        """Cancel all orders filtered by buy or sell orders"""
        orders = self.get_active_orders()
        for order in orders:
            if sell_orders and order.side == "sell":
                self.cancel_order(order)
            elif buy_orders and order.side == "buy":
                self.cancel_order(order)

    def create_file_from_add_trades_function(
        self, all_trades, signifier=None, full_file=False
    ):
        """create file from the trades that we get in the Trade object type.

        Args:
            all_trades (List of Trade type object): Trade type object organized
            signifier (string, optional): unique file signifier. Defaults to None.
            full_file (bool, optional): if its a full file we don't include "tst" in the file name. Defaults to False.

        Returns:
            _type_: _description_
        """
        many_trades = pd.DataFrame()
        trades = []
        for idx, period_trades in enumerate(all_trades):
            if period_trades != []:
                trades = pd.DataFrame.from_records(
                    [trade.to_dict() for trade in period_trades]
                )
                symbol = period_trades[0].symbol
                many_trades = many_trades.append(trades)
        if full_file:
            name = symbol + "_binance" + ".csv"
            many_trades.to_csv(name)
        else:
            if signifier:
                name = symbol + "_binance_tst" + signifier + ".csv"
                many_trades.to_csv(name)
            else:
                name = symbol + "_binance_tst" + ".csv"
                many_trades.to_csv(name)
        return name

    def helper_find_in_file(self, start_date_iso, end_date_iso, symbol):
        file_name = symbol + '.csv'
        start_time_obj = start_date_iso
        now_obj = end_date_iso
        trade_byte_start = self.helper_binary_find_target_time_iso(file_name, start_time_obj, False)
        trade_byte_end = self.helper_binary_find_target_time_iso(file_name, now_obj, True)
        return trade_byte_start, trade_byte_end

    def helper_binary_find_target_time_iso(self, file, target, after=None):
        f = open(file, 'r')
        file_size = os.fstat(f.fileno()).st_size
        #middle_pos = file_size // 2
        #f.seek(middle_pos)
        #line = f.readline()
        left = 0
        right = file_size
        compare_me = None
        if isinstance(target, datetime.datetime):
            target=datetime.datetime.isoformat(target)
            if '+00:00' not in target:
                target+='+00:00'
            target=datetime.datetime.fromisoformat(target)
        if not isinstance(target, datetime.datetime):
            target = datetime.fromisoformat(target)
        while left < right:
            mid = (left + right) // 2
    # Read the next line from the file
            f.seek(mid)
            while f.read(1) != '\n' and f.tell() > 0:
                f.seek(f.tell() - 2, os.SEEK_SET)
            capture_f_position = f.tell()
            line = f.readline().split(',')
            while f.read(1) != '\n' and f.tell() > 0:
                f.seek(f.tell() - 2, os.SEEK_SET)
            capture_f_position1 = f.tell()
            line1 = f.readline().split(',')
            compare_me = datetime.datetime.fromisoformat(line[2])
            compare_me1 = datetime.datetime.fromisoformat(line1[2])

            if compare_me <= target and compare_me1 >= target:
                if after == True:
                    return [line1, capture_f_position1]
                else:
                    return [line, capture_f_position]
            elif target > compare_me and target > compare_me1:
                left = mid + 1
            else:
                right = mid - 1
            human_see = datetime.datetime.isoformat(target)
            human_see1 = datetime.datetime.isoformat(compare_me)
            hello = human_see1 >= human_see
            ohhhh = None
        return -1

    def generic_recent_trades(self, symbol):
        endpoint = '/api/v3/trades'
        data = {'symbol': symbol, 'limit': 1}
        most_recent_trade = self.get_rest(request=endpoint, data=data)
        most_recent_trade = most_recent_trade[0]
        # most_recent_trade = json.loads(most_recent_trade)
        all_keys = most_recent_trade.keys()
        for key in all_keys:
            if key == 'price':
                return Decimal(most_recent_trade[key])

    def get_current_price(self, symbol):
        current_price = self.generic_recent_trades(symbol=symbol)
        return current_price

    def search_for_id(self, target_id, filename):
        file_o = open(filename, 'r')
        file_o.seek(os.fstat(file_o.fileno()).st_size)
        while file_o.tell() > 0 and file_o.read(1) != '\n':
            file_o.seek(file_o.tell() - 2, os.SEEK_SET)
        file_o.readline()
        end_byte_line = file_o.tell()
        left = 0
        right = os.fstat(file_o.fileno()).st_size
        return_bool = False
        while left < right:
            mid = (left + right) // 2
            file_o.seek(mid)
            while file_o.tell() > 0 and file_o.read(1) != '\n':
                file_o.seek(file_o.tell() - 2, os.SEEK_SET)
            line = file_o.readline()
            try:
                viewed_id = int(line.split(',')[1])
            except:
                return False
            if target_id == viewed_id:
                return_bool = True
                file_o.close()
                return return_bool
            if target_id < viewed_id:
                right = mid + 1
            elif target_id > viewed_id:
                left = mid - 1
            if left >= os.fstat(file_o.fileno()).st_size or right <= 0:
                file_o.close()
                return False
            if viewed_id + 1 == target_id:
                file_o.close()
                return True
        file_o.close()
        return return_bool

    def find_if_ids_exist(self, filename, beginning_id, end_id):
        try:
            f = open(filename, 'r')
            f.close()
        except:
            return False, False
        beginning = self.search_for_id(beginning_id, filename)
        end = self.search_for_id(end_id, filename)
        return beginning, end

    def create_file_with_trades(self, symbol, beginning_id, end_id, ema_periods=None):
        all_missing = self.get_trades_raw(symbol=symbol, from_id=beginning_id, to_id=end_id)
        all_missing_df = pd.DataFrame()
        trades = pd.DataFrame.from_records([trade.to_dict() for trade in all_missing])
        all_missing_df = all_missing_df.append(trades)
        # convert to file since we don't already have one
        all_missing_df.to_csv(
            symbol + '.csv', header=True, mode="w", line_terminator="\n"
        )

    def download_symbol(self, start_time_iso, end_time_iso, symbol):
        start_time = self.date_to_milliseconds(start_time_iso)
        end_time = self.date_to_milliseconds(end_time_iso)
        print(len(str(start_time)))
        print(len(str(end_time)))
        beginning_data = {
            "symbol": symbol,
            "startTime": start_time,
            "limit": 1,
        }
        end_data = {
            "symbol": symbol,
            "startTime": end_time,
            "limit": 1,
        }
        beginning_id = self.get_rest("/api/v3/aggTrades", data=beginning_data)
        beginning_id = beginning_id[0]["a"]
        end_id = self.get_rest("/api/v3/aggTrades", data=end_data)[0]["a"]
        symbol_file = symbol + '.csv'
        end, beginning = self.find_if_ids_exist(symbol_file, beginning_id=beginning_id, end_id=end_id)
        if not end or not beginning:
            self.create_file_with_trades(symbol=symbol, beginning_id=beginning_id, end_id=end_id)

    def find_in_file(self, candles, file_name, index_of_id=1, tst=False, start_date = None, end_date = None, symbol = None):
        """function to give indicies of initial and final index of the candles given of all trades in
        between the two.

        Args:
            candles (list of Candle object): candle objects contain dates which is what is primarily used here
            file_name (string): since aggregating trades is expensive, we look it up in a file
            index_of_id (int, optional): files may change in the future but right now they are in the second spot of the file
            . Defaults to 1.
            tst (bool, optional): used for debugging purposes. Defaults to False.

        Returns:
            int,int: initial and final index of first candle and last candle.
        """
        # get our start and endd dates based on the first and last candle
        if not candles and start_date and end_date and symbol:
            start_date = self.date_to_milliseconds(start_date)
            end_date = self.date_to_milliseconds(end_date)
            symbol = symbol
        elif candles:
            wut = candles[0].timestamp
            if isinstance(candles[0].timestamp, datetime.datetime):
                start_date = candles[0].timestamp.isoformat()
                start_date = self.date_to_milliseconds(start_date)
            else:
                start_date = candles[0].timestamp
            if isinstance(candles[-1].timestamp, datetime.datetime):
                end_date = candles[-1].timestamp.isoformat()
                end_date = self.date_to_milliseconds(end_date)
            else:
                end_date = candles[-1].timestamp
            symbol = candles[0].symbol
        else:
            return -1
        # since the dates are based of the structure of Candle class they need to converted to ms for binance api
        # initially set our outut variables to none
        initial_index = None
        final_index = None
        file = False
        all_lines = []
        ready_to_read = False
        # beginning and end data are used for the get request from the binance api
        beginning_data = {
            "symbol": symbol,
            "startTime": start_date,
            "limit": 1,
        }
        end_data = {"symbol": symbol, "startTime": end_date, "limit": 1}
        # beginning_id and end_id are found so that we can see what ids inbetween we need
        beginning_id = self.get_rest("/api/v3/aggTrades", data=beginning_data)[0]["a"]
        end_id = self.get_rest("/api/v3/aggTrades", data=end_data)[0]["a"] - 1
        # elaborate way to make sure the file exists and opens
        if file_name is not None:
            try:
                f = open(file_name, "r")
                file = True
            except:
                print("unable to open file")
                file = False
        # make sure there is always a symbol that binance recognizes in the file_name
        if file:
            if symbol in file_name:
                ready_to_read = True
            else:
                return -1, -1
        if ready_to_read:
            all_lines = f.readlines()
            if beginning_id > end_id:
                end_id = beginning_id
            initial_index, final_index = self.get_indicies_for_file(
                all_lines, beginning_id, end_id, index_of_id
            )
        # if we have an empty file we need to create one and then send the indicies of the beginning and end id back
        if all_lines == []:
            all_missing_df = pd.DataFrame()
            # get_trades_raw deals with getting all the trades between two points in time or two dates
            all_missing = self.get_trades_raw(
                symbol=symbol, from_id=beginning_id, to_id=end_id + 1
            )
            trades = pd.DataFrame.from_records(
                [trade.to_dict() for trade in all_missing]
            )
            all_missing_df = all_missing_df.append(trades)
            # convert to file since we don't already have one
            all_missing_df.to_csv(
                str(file_name), header=True, mode="w", line_terminator="\n"
            )
            final_file = open(str(file_name), "r")
            all_lines = final_file.readlines()
            final_file.close()
            # return the indicies
            return self.get_indicies_for_file(
                all_lines=all_lines,
                beginning_id=beginning_id,
                end_id=end_id,
                index_of_id=index_of_id,
                print_me=False,
            )
        # if the final_inex is a number and the initial_index is a number that means we basically already found them
        if final_index is not None and initial_index is not None:
            return initial_index, final_index
        # if we have a final index but no initial index, it means we are missing trades at the beginning and need to findd them
        if final_index is not None and initial_index is None:
            # common_id is the id that we start having so we need to get all trades up this id
            common_id = all_lines[1].split(",")[index_of_id]
            missing_candles = []
            all_missing_df = pd.DataFrame()
            # get all the trades up to the common id
            all_missing = self.get_trades_raw(
                symbol=candles[0].symbol, from_id=beginning_id, to_id=common_id
            )
            trades = pd.DataFrame.from_records(
                [trade.to_dict() for trade in all_missing]
            )
            all_missing_df = all_missing_df.append(trades)
            # we are overwriting what is in the file because we don't want to make a new one for efficiency and because we still have all the
            # lines that were in the file saved to all_lines
            all_missing_df.to_csv(file_name, header=True, line_terminator="\n")
            f = open(file_name, "a")
            # since we overwrote the file we need to add what was in it before to the end
            for idx, line in enumerate(all_lines):
                # the first line is the header so we don't want to add it
                if idx > 0 and line != "\n":
                    f.write(line)
            f.close()
            # we do this so the cursor is pointing to the first line instead of the last line, otherwise readlines() doesn't work correctly
            final_file = open(file_name, "r")
            all_lines = final_file.readlines()
            final_file.close()
            return self.get_indicies_for_file(
                all_lines=all_lines,
                beginning_id=beginning_id,
                end_id=end_id,
                index_of_id=index_of_id,
                print_me=False,
            )
        # if we don't have a final index and do have an initial index we are missing trades at the end
        if final_index is None and initial_index is not None:
            # the id that we need to get after is the last one
            common_id = all_lines[-1].split(",")[index_of_id]
            all_missing_df = pd.DataFrame()
            end_id = end_id + 1
            # put all the missing traded from get_trades_raw into all_missing
            all_missing = self.get_trades_raw(
                symbol=candles[0].symbol, from_id=common_id, to_id=end_id
            )
            trades = pd.DataFrame.from_records(
                [trade.to_dict() for trade in all_missing]
            )
            all_missing_df = all_missing_df.append(trades)
            print("all_missing", all_missing_df)
            # since we aren't adding to the beginning we can simply be in append mode which sets the curser to the end of the file
            all_missing_df.to_csv(file_name, header=False, mode="a")
            # reset curser to the beginning of the file
            final_file = open(file_name, "r")
            all_lines = final_file.readlines()
            final_file.close()
            return self.get_indicies_for_file(
                all_lines=all_lines,
                beginning_id=beginning_id,
                end_id=end_id - 1,
                index_of_id=index_of_id,
                print_me=True,
            )
        if final_index is None and initial_index is None:
            all_missing_df = pd.DataFrame()
            # get_trades_raw deals with getting all the trades between two points in time or two dates
            all_missing = self.get_trades_raw(
                symbol=symbol, from_id=beginning_id, to_id=end_id + 1
            )
            if all_missing != None:
                trades = pd.DataFrame.from_records(
                    [trade.to_dict() for trade in all_missing]
                )
                all_missing_df = all_missing_df.append(trades)
                # convert to file since we don't already have one
                all_missing_df.to_csv(
                    str(file_name), header=True, mode="w", line_terminator="\n"
                )
                final_file = open(str(file_name), "r")
                all_lines = final_file.readlines()
                final_file.close()
                # return the indicies
                return self.get_indicies_for_file(
                    all_lines=all_lines,
                    beginning_id=beginning_id,
                    end_id=end_id,
                    index_of_id=index_of_id,
                    print_me=False,
                )

    def get_indicies_for_file(
        self, all_lines, beginning_id, end_id, index_of_id, print_me=False
    ):
        """Helper function for find_in_file since it returnes indicies of a file.

        Args:
            all_lines (list of string): raw input from file.readlines()
            beginning_id (int): id to look for in all_lines initially
            end_id (int): id to stop looking for in all_lines
            index_of_id (int): files may change in the future so the index of the id when doing all_lines.split(',')
            print_me (bool, optional): print out all_lines for debugging purposes. Defaults to False.

        Returns:
            int,int: initial and final indicies of beginning and end id
        """
        initial_index, final_index = None, None
        # for every line we want to check if the id present in the file is equal to one of the ids we gave the function
        for idx, line in enumerate(all_lines):
            list_of_values = line.split(",")
            id_in_line = list_of_values[index_of_id]
            if id_in_line == str(beginning_id):
                initial_index = idx
            if id_in_line == str(end_id):
                final_index = idx
        if print_me == True:
            print("all_lines", all_lines)
        return initial_index, final_index

    def get_trades_raw(
        self,
        symbol,
        max_trades=1000,
        start_date=None,
        end_date=None,
        from_id=None,
        to_id=None,
    ):
        """get_trades_raw gets all trades in the trade class format from the binance.us api from a start date to an end date OR a from a start id
        to an end id.

        Args:
            symbol (string): string of the symmbol we getting trades for for the request function
            max_trades (int, optional): the batch size of trades to get, the max for binance is 1000. Defaults to 1000.
            start_date (string, optional): date in ms to start at to get trades. Defaults to None.
            end_date (string, optional): date in ms to end at to get trades_. Defaults to None.
            from_id (int, optional): known id to start from. Defaults to None.
            to_id (int, optional): known id to end at. Defaults to None.

        Returns:
            List of Trade object: Trade object simplifies the trades which are vague in the json format undervisual
            scrutiny. See https://docs.binance.us/#get-historical-trades for api info.
        """
        # make sure we are only using from and to id if the user doesn't want a date span
        if from_id is not None and to_id is not None and start_date is None and end_date is None:
            original_data = {"symbol": symbol, "fromId": from_id, "limit": max_trades}
            # first get some data from the api concerning the data provided above
            response = self.get_rest("/api/v3/aggTrades", data=original_data)
            final_idx = 0
            temp_response = response
            final_response = []
            found = False
            # loop until we have found the id we are looking for
            while not found:
                final_id_of_response = temp_response[-1]["a"]
                # if the last element id is greater than or equal to the to_id we only want to grab up until that id and not more
                if temp_response[-1]["a"] >= int(to_id):
                    # for every trade in json check if the id element or trade['a'] is the one we ant
                    for idx, trade in enumerate(temp_response):
                        if trade["a"] == int(to_id):
                            found = True
                            final_idx = idx
                    # only grab up until the to_id
                    final_response += temp_response[:final_idx]
                    # request_agg_response_to_trade converts all the json trades we have agglomerated to trade objects for organization
                    trades = self.request_agg_response_to_trade(symbol, final_response)
                    return trades
                else:
                    final_response += temp_response
                    temp_data = {
                        "symbol": symbol,
                        "fromId": int(temp_response[-1]["a"]),
                        "limit": 1000,
                    }
                    temp_response = self.get_rest("/api/v3/aggTrades", data=temp_data)
                    hello = "hello"
        else:
            trades = []
            # fill in default parameters for num_trades, start_date, and end_date
            if start_date:
                start_date = self.date_to_milliseconds(start_date)
                if end_date:
                    end_date = self.date_to_milliseconds(end_date)
                else:
                    end_date = int((time.time() + 60) * 1000)
                terminal_date = end_date
                data = {
                    "symbol": symbol,
                    "startTime": start_date,
                    "endTime": end_date,
                    "limit": max_trades,
                }
                # first get some data from the api concerning the data provided above
                response = self.get_rest("/api/v3/aggTrades", data=data)
                # this data gets the id that we should be ending at because we need to know where to end
                data = {"symbol": symbol, "startTime": end_date, "limit": 1}
                # the request that actually gets the data from the data above
                end_id = self.get_rest("/api/v3/aggTrades", data=data)
                # the ids are uner the 'a' element of the request so we use this to get the finishing id needed
                end_id = end_id[0]["a"] - 1
                found = False
                # if we know that the response is less than our max that means we don't need to loop around get more
                if len(response) < max_trades:
                    found = True

                while found != True:
                    data = {
                        "symbol": symbol,
                        "fromId": response[-1]["a"],
                        "limit": 1000,
                    }
                    # look through every thousand trades to see if we hit our end id.
                    temp_response = self.get_rest("/api/v3/aggTrades", data=data)
                    for idx, id in enumerate(temp_response):
                        # if we did hit our end id then we only want to return the aggregate response + the trades to the end id
                        if temp_response[idx]["a"] == end_id + 1:
                            response += temp_response[:idx]
                            found = True
                            trades = self.request_agg_response_to_trade(
                                symbol, response
                            )
                            return trades
                    # if we didn't hit our end_id we haven't gotten where we want and just add all the trades to our list
                    else:
                        response += temp_response
                trades = self.request_agg_response_to_trade(symbol, response)
                return trades

    def request_agg_response_to_trade(self, symbol, response):
        """converts raw json trades to Trade objects

        Args:
            symbol (string): symbol of the response
            response (json list): raw json trades

        Returns:
            List of trade objects: trade objects are far more organized than raw trades
        """
        trades = []
        # for every json trade element we want to organize it into an object for Trade
        for trade in response:
            trades.append(
                Trade(
                    symbol,
                    self.convert_time(trade["T"]),
                    trade["p"],
                    trade["q"],
                    ("buy" if trade["m"] else "sell"),
                    id=trade["a"],
                )
            )
        return trades

    def get_trades(self, symbol, max_trades=1000, start_date=None, end_date=None):
        """Get trades for the given symbol at a given time
        symbol = symbol to get trades for
        max_trades = maximum trades to get for a given period, uses start and end date if None
        start_date = timestamp to start getting trades, defaults to max_trades before end date if None
        end_date = timestamp to stop getting trades, defaults to current date if None and start_date is None
        """
        trades = []
        # fill in default parameters for num_trades, start_date, and end_date
        if start_date:
            start_date = self.date_to_milliseconds(start_date)
            if end_date:
                end_date = self.date_to_milliseconds(end_date)
            else:
                end_date = int((time.time() + 60) * 1000)
            terminal_date = end_date
            while True:
                end_date = start_date + 3600000
                data = {"symbol": symbol, "startTime": start_date, "endTime": end_date}
                response = self.get_rest("/api/v3/aggTrades", data=data)
                for trade in response:
                    if max_trades and len(trades) >= max_trades:
                        return trades
                    trades.append(
                        Trade(
                            symbol,
                            trade["T"],
                            trade["p"],
                            trade["q"],
                            ("buy" if trade["m"] else "sell"),
                            id=trade["a"],
                        )
                    )
                start_date = end_date + 1
                if end_date > terminal_date:
                    return trades
        if not start_date:
            if not end_date:
                end_date = int((time.time() + 60) * 1000)
            end_date = date_to_milliseconds(end_date)
            terminal_date = 1230768000000
            while True:
                start_date = end_date - 3600000
                data = {"symbol": symbol, "startTime": start_date, "endTime": end_date}
                response = self.get_rest("/api/v3/aggTrades", data=data)
                for trade in response:
                    if max_trades and len(trades) >= max_trades:
                        return trades
                    trades.append(
                        Trade(
                            symbol,
                            trade["T"],
                            trade["p"],
                            trade["q"],
                            ("buy" if trade["m"] else "sell"),
                            id=trade["a"],
                        )
                    )
                end_date = start_date - 1
                if start_date < terminal_date:
                    return trades

    def get_candles(self, symbol, num_candles=20, period=5):
        """Get candles for the given symbol at the given increment"""
        if num_candles == 0:
            return []
        periods = {
            1: "1m",
            3: "3m",
            5: "5m",
            15: "15m",
            30: "30m",
            60: "1h",
            120: "2h",
            240: "4h",
            360: "6h",
            480: "8h",
            720: "12h",
            1440: "1d",
            4320: "3d",
            10080: "1w",
            43200: "1M",
        }
        if period not in periods:
            self.add_msg("Given candle period is not in the list of periods")
            return []
        period_str = periods[period]

        # End time defined by the current server time. Get the last period it would have reported
        end_time = int(time.time() * 1000)
        end_time -= end_time % (period * 60000)

        max_limit = 1000 if num_candles > 1000 else num_candles
        start_time = end_time - (period * 60000 * (num_candles - 1))

        candles_returned = 0
        candles = []

        while 1:
            response = self.get_rest(
                "/api/v3/klines",
                data={
                    "symbol": symbol,
                    "interval": period_str,
                    "limit": max_limit,
                    "startTime": start_time,
                },
            )
            for candle in response:
                candles.append(
                    Candle(
                        symbol,
                        self.convert_time(candle[0]),
                        first=candle[1],
                        last=candle[4],
                        low=candle[3],
                        high=candle[2],
                        volume_base=candle[5],
                        volume_quote=candle[7],
                        period_len=period,
                    )
                )
                candles_returned += 1
                if candles_returned >= num_candles:
                    break
            start_time = int(candles[-1].timestamp.timestamp() + (60 * period)) * 1000
            if (
                candles_returned >= num_candles
                or len(response) < max_limit
                or start_time > end_time
            ):
                break

        return candles

    def get_last_candles(self, symbol, num_prev_candles, len_period_minutes):
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=(num_prev_candles + 5)*(len_period_minutes)*60)
        start_date_iso = start_time.isoformat()
        end_date_iso = end_time.isoformat()
        last_candles = self.get_candles_on_timestamps(symbol=symbol, start_date_iso=start_date_iso, end_date_iso=end_date_iso)
        return last_candles[-num_prev_candles:]

    def periods_passed_now_timestamp(self, timestamp_iso, period_len_minutes):
        now = datetime.datetime.now()
        past = datetime.fromisoformat(timestamp_iso)
        total_seconds = (now - past).total_seconds()
        total_periods_passed = int(total_seconds/(period_len_minutes*60))
        return total_periods_passed

    def get_supertrend(self, backup):
        # TODO: implement this
        pass

    def set_dict(self, myDict):
        #empty function that is supposed to be empty because its only something the sim uses
        #bad practice but this is the only one I can forsee in efforts to keep exchanges essentially the same
        pass

    def get_candles_on_timestamps(self, symbol, start_date_iso, end_date_iso, period_len_minutes=5):
        candles = []
        periods = {
            1: "1m",
            3: "3m",
            5: "5m",
            15: "15m",
            30: "30m",
            60: "1h",
            120: "2h",
            240: "4h",
            360: "6h",
            480: "8h",
            720: "12h",
            1440: "1d",
            4320: "3d",
            10080: "1w",
            43200: "1M",
        }
        period_str = periods[period_len_minutes]
        if start_date_iso:
            start_date = self.date_to_milliseconds(start_date_iso)
        if end_date_iso:
            end_date = self.date_to_milliseconds(end_date_iso)
        response = self.get_rest(
            "/api/v3/klines",
            data={
                "symbol": symbol,
                "interval": period_str,
                "startTime": start_date,
                "endTime": end_date,
            },
        )
        for candle in response:
            candles.append(
                Candle(
                    symbol,
                    (self.convert_time(candle[0])).isoformat(),
                    first=candle[1],
                    last=candle[4],
                    low=candle[3],
                    high=candle[2],
                    volume_base=candle[5],
                    volume_quote=candle[7],
                    period_len=period_len_minutes,
                )
            )

        return candles

    def add_trades_to_candles(self, candles):
        """given a candles list goes through and gets all the trades from the candles timestamp to the next
        candles timestamp and aggregates them into a return list. While it never technically "adds" to candles
        it completes the vision of having corresponding indicies to candles object so it is easy to track

        Args:
            candles (list of Candle object): list of Candle object that give easy access to 'high', 'low',
            of a given period

        Returns:
            list of list of Trade object: each index is a list of all the trades that happened during the timestamp
            starting at the timestamp of the candle and going through the start of the next candle
        """
        all_candles_trades = []
        for idx, candle in enumerate(candles):
            # make sure we never go out of bounds
            if idx + 1 <= len(candles) - 1:
                # this_candle_trades is a list of Trade objects
                this_candle_trades = self.get_trades_raw(
                    candle.symbol,
                    max_trades=1000,
                    start_date=candle.timestamp,
                    end_date=candles[idx + 1].timestamp,
                )
                # aggregate the list of Trade objects where the index is the index of the candles list
                all_candles_trades.append(this_candle_trades)
        return all_candles_trades

    def replace_order(self, order, new_id=None):
        order.last_time_modified = datetime.utcnow()
        self.prepare_order(order=order)
        """Replace a previous order with new parameters"""
        if order.symbol_obj.prices_equal(order.price, 0):
            self.add_msg("Error: attempt to place order at price of 0")
            return None
        elif order.symbol_obj.volumes_equal(order.total_amt, 0):
            self.add_msg("Error: attempt to place order at volume of 0")
            return None
        print(
            "Replacing",
            order.symbol_obj.name,
            "id:",
            order.assigned_id,
            "with new",
            order.side,
            "order at price of",
            order.symbol_obj.string_price(order.price),
            "and volume of",
            order.symbol_obj.string_volume(order.total_amt),
        )
        self.cancel_order(order)
        # if new_id:
        #     order.order_id = new_id
        # else:
        #     order.order_id = None
        return self.new_order(order)

    def get_account_trades(self, number=100, symbol=None, start=None):
        """Get recent trades for this account"""
        data = {"limit": number, "symbol": symbol.name}
        if start:
            data["startTime"] = date_to_milliseconds(start)
        if symbol:
            data["symbol"] = symbol.name
            response = self.get_rest("/api/v3/myTrades", data=data, requires_auth=True)
            trades = []
        else:
            return []
        for trade in response:
            trades.append(
                Trade(
                    self.get_symbol_from_name(trade["symbol"]),
                    self.convert_time(trade["time"]),
                    price=trade["price"],
                    volume=trade["qty"],
                    fee=trade["commission"],
                    side="buy" if trade["isBuyer"] else "sell",
                    id=trade["id"],
                    order_id=trade["orderId"],
                )
            )

        return trades

    def get_account_orders(self, number=100, symbol=None, order_id=None, start=None):
        """Get recent orders for this account"""
        data = {"limit": number, "symbol": symbol.name}
        if start:
            data["startTime"] = date_to_milliseconds(start)
        if order_id:
            data["orderId"] = order_id
        orders = []
        response = self.get_rest("/api/v3/allOrders", data=data, requires_auth=True)
        for order in response:
            order_symbol = self.get_symbol_from_name(order["symbol"])
            time = self.convert_time(order["time"])
            orders.append(
                Order(
                    order_symbol,
                    time,
                    order_id=order["orderId"],
                    price=order["price"],
                    volume=order["origQty"],
                    executed_volume=order["executedQty"],
                    side=order["side"],
                )
            )
        return orders

    def get_transaction(self, transaction_id):
        """Get transaction info."""
        return self.get_rest("/account/transactions/" + transaction_id)

    def on_open(self):
        self.add_msg("Socket started", "socket_msg")
        pass

    def on_socket_close(self):
        self.add_msg("Socket closed", "socket_msg")
        if time.time() - self.last_socket_update < 60:
            return
        if self.ws_streams["account"]:
            self.ws_streams["account"] = None
        self.init_callbacks()

    def on_socket_message(self, message, timestamp):
        if "stream" not in message or message["stream"] == "":
            print('ERROR')
            print(message)
            return

        self.add_msg("socket message:" + str(message))
        print('socket_message', message)

        stream = message["stream"]
        data = message["data"]
        print( "@" not in stream)
        print( list(self.ws_streams["account"].keys())[-1] == stream)
        if "@" not in stream and list(self.ws_streams["account"].keys())[-1] == stream:
            # Message is for the account
            # Check if this is an order update message
            if data["e"] == "executionReport":
                self.order_executed_handler(data)

    def order_executed_handler(self, data):
        if (data["o"] == "MARKET" and data['X'] == 'FILLED') or (data["o"] == 'MARKET' and data['X'] == "PARTIALLY_FILLED"):
            symbol = self.get_symbol_from_name(data["s"])
            order = Order(
                symbol,
                timestamp=data["O"],
                order_id=data["c"],
                price=data["L"],
                volume=data["q"],
                side=data["S"],
                completed=True if data["X"] == "FILLED" else False,
                executed_volume=data["z"],
            )
            trade = Trade(
                symbol,
                data["T"],
                price=data["L"],
                volume=data["l"],
                id=data['t'],
                side=data["S"],
                fee=data["n"],
            )
            self.add_msg("order executed handler: " + str(data))
            threading.Thread(
                target=self.on_market_trade,
                args=(
                    order,
                    trade,
                ),
            ).start()

        if data["x"] == "TRADE" and data["o"] == "LIMIT":
            symbol = self.get_symbol_from_name(data["s"])
            order = Order(
                symbol,
                timestamp=data["O"],
                order_id=data["c"],
                price=data["L"],
                volume=data["q"],
                side=data["S"],
                completed=True if data["X"] == "FILLED" else False,
                executed_volume=data["z"],
            )
            trade = Trade(
                symbol,
                data["T"],
                price=data["L"],
                volume=data["l"],
                id=data['t'],
                side=data["S"],
                fee=data["n"],
            )
            self.add_msg("order executed handler: " + str(data))
            threading.Thread(
                target=self.on_order_executed,
                args=(
                    order,
                    trade,
                ),
            ).start()

    def on_socket_error(self, error):
        self.add_msg(str(error), "socket_error")

    def socket_send(self, data, timeout=10, callback=None):
        pass

    def keep_alive(self):
        while self.running:
            for i in range(3600):  # Ping every 30 minutes
                if not self.running:
                    return
                time.sleep(0.5)

            listen_key = self._get_listen_key()
            if listen_key:
                data = {"listenKey": listen_key}
                response = self.get_rest(
                    "/api/v3/userDataStream",
                    request_type="PUT",
                    data=data,
                    requires_auth=False,
                )
            self.add_msg("keep alive response: " + str(response))

    def init_callbacks(self, symbols=[]):
        """Code to initialize callback functions if necessary"""
        self.last_socket_update = time.time()
        if self.on_order_executed and (
            "account" not in self.ws_streams or not self.ws_streams["account"]
        ):
            # Get a listen key
            listen_key_data = self.get_rest(
                "/api/v3/userDataStream", request_type="POST", requires_auth=False
            )
            listen_key = listen_key_data["listenKey"]

            # Start the web socket
            endpoint = WebSocketConnectorThread(
                self.ws_endpoint + "/stream?streams=" + listen_key,
                self.on_socket_message,
                self.on_socket_close,
                self.on_socket_error,
                self.on_open,
            )
            endpoint.start()
            self.ws_streams["account"] = {listen_key: endpoint}

            # Start the keep alive thread
            self.keep_alive_thread = threading.Thread(target=self.keep_alive)
            self.keep_alive_thread.start()

        for symbol in symbols:
            pass

    def refresh(self):
        super().refresh()

        self.rest_session = requests.session()
        if self.auth:
            self.rest_session.headers.update(
                {
                    "Accept": "application/json",
                    "User-Agent": "binance/python",
                    "X-MBX-APIKEY": self.public_key,
                }
            )

        # Populate the currencies
        data = self.get_rest("/api/v1/exchangeInfo")
        self.currencies = set()
        self.symbols = []
        for symbol in data["symbols"]:
            if symbol["status"] != "TRADING":
                continue
            price_filter = [
                value
                for value in symbol["filters"]
                if value["filterType"] == "PRICE_FILTER"
            ][0]
            lot_filter = [
                value
                for value in symbol["filters"]
                if value["filterType"] == "LOT_SIZE"
            ][0]
            min_notional = [
                value
                for value in symbol["filters"]
                if value["filterType"] == "MIN_NOTIONAL"
            ][0]
            self.symbols.append(
                Symbol(
                    name=symbol["symbol"],
                    base=symbol["baseAsset"],
                    quote=symbol["quoteAsset"],
                    base_increment=lot_filter["stepSize"],
                    quote_increment=price_filter["tickSize"],
                    min_value=min_notional["minNotional"],
                )
            )
            self.currencies.add(symbol["baseAsset"])
            self.currencies.add(symbol["quoteAsset"])
        self.currencies = list(self.currencies)

    def _get_listen_key(self):
        if "account" in self.ws_streams and len(self.ws_streams["account"].keys()) > 0:
            return list(self.ws_streams["account"].keys())[-1]
        return None

    def exit(self):
        """Perform necessary steps to exit the program"""
        print("Exiting...")
        self.running = False