import datetime
import json
import logging
import random
import time
import uuid
from decimal import ROUND_HALF_UP, Decimal
from queue import Queue

import dateutil.parser
import pytz
import requests


def date_to_timestamp(timestamp):
    if type(timestamp) is str:
        return dateutil.parser.parse(timestamp)
    else:
        return timestamp

def date_to_milliseconds(date_str):
    """Convert UTC date to milliseconds
    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/
    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = date_to_timestamp(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)

class Level:
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

class Symbol:
    TOLERANCE=0.00000000001
    def __init__(self, name="BTCUSD", base="BTC", quote="USD", base_increment=0.00001, quote_increment=0.01, min_value = 0.0):
        self.name = name
        self.base = base
        self.quote = quote
        self.base_increment = float(base_increment)
        self.quote_increment = float(quote_increment)
        self.min_value = float(min_value)

    def buyable(self, price, funds_available, fee):
        """Determine the maximum possible buyable volume given funds available"""
        amount = self.round_volume(funds_available / self.round_price(price) * (1.0 - fee))
        #print('Buying', amount, self.base, 'at price of', price, self.quote, '(' + str(funds_available), self.quote + ')')
        return amount

    def round_volume(self, volume, round_up=False):
        """Round volume up or down to nearest increment"""
        rounded = int(volume / self.base_increment + self.TOLERANCE) * self.base_increment
        return rounded + self.base_increment if round_up else rounded

    def round_price(self, price, round_up=False):
        """Round price up or down to nearest increment"""
        rounded = int(price / self.quote_increment + self.TOLERANCE) * self.quote_increment
        return rounded + self.quote_increment if round_up else rounded

    def string_price(self, price):
        if self.quote_increment + self.TOLERANCE < 1:
            return format(Decimal(price).quantize(Decimal(str(self.quote_increment)), ROUND_HALF_UP), 'f')
        else:
            return str(round(price / round(self.quote_increment)) * round(self.quote_increment))

    def string_volume(self, volume):
        if self.base_increment + self.TOLERANCE < 1:
            return format(Decimal(volume).quantize(Decimal(str(self.base_increment)), ROUND_HALF_UP), 'f')
        else:
            return str(round(volume / round(self.base_increment)) * round(self.base_increment))

    def quote_if_sold(self, price, volume, fee):
        """Calculates the exact currency you would have if sold at the given price and fee"""
        amount = self.round_price(self.round_price(price) * self.round_volume(volume) * (1.0 - fee))
        #print('Selling', volume, self.base, 'at price of', price, '(' + str(amount), self.quote + ')')
        return amount

    def verify(self, price, volume):
        if float(price) * float(volume) + self.TOLERANCE > self.min_value:
            return True
        return False

    def prices_equal(self, price1, price2):
        """Determines if two prices are equal"""
        return self.string_price(float(price1)) == self.string_price(float(price2))

    def volumes_equal(self, volume1, volume2):
        """Determines if two volumes are equal"""
        return self.string_volume(float(volume1)) == self.string_volume(float(volume2))

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return 'Symbol - ' + self.base + ' to ' + self.quote

class Trade:
    def __init__(self, symbol, timestamp, price=0.0, volume=0.0, side='buy', fee=0.0, id=None, order_id=None):
        self.symbol = symbol
        self.timestamp = date_to_timestamp(timestamp)
        self.price = float(price)
        self.volume = float(volume)
        self.side = side.lower()
        self.fee = float(fee)
        self.order_id = order_id
        # Assume id is always an int (if it exists)
        if id:
            self.id = int(id)
        else:
            self.id = None

    def __copy__(self):
        return Trade(self.symbol, self.timestamp, price=self.price, volume=self.volume, side=self.side, fee=self.fee, id=self.id, order_id=self.order_id)

    def __repr__(self):
        return 'Trade (' + self.side + ' ' + self.symbol + ' at price ' + str(self.price) + ' and volume ' + str(self.volume) + ' fee: ' + str(self.fee) + ' id: ' + str(self.id) + ')'
    def to_dict(self):
        return {
            'id':self.id,
            'timestamp':self.timestamp,
            'symbol':self.symbol,
            'price':self.price,
            'buy_or_sell':self.side,
            'volume':self.volume,
        }
class Candle:
    def __init__(self, symbol, timestamp, first=0.0, last=0.0, low=0.0, high=0.0, volume_base=0.0, volume_quote=0.0, period_len=60, name = None):
        self.symbol = symbol
        self.timestamp = date_to_timestamp(timestamp)
        self.first = float(first)
        self.last = float(last)
        self.low = float(low)
        self.high = float(high)
        self.volume_base = float(volume_base)
        self.volume_quote = float(volume_quote)
        self.period_len = int(period_len)
        self.name = name

    def copy(self):
        return Candle(self.symbol, self.timestamp, first=self.first, last=self.last, low=self.low, high=self.high, volume_quote=self.volume_quote, volume_base=self.volume_base, period_len=self.period_len)

    def __repr__(self):
        return 'Candle (O ' + str(self.first) + ' H ' + str(self.high) + ' L ' + str(self.low) + ' C ' + str(self.last) + ' time: ' + self.timestamp.isoformat() + ')'

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'open': self.first,
            'high': self.high,
            'low': self.low,
            'close': self.last,
            'volume': self.volume_base,
            'volumeQuote': self.volume_quote,
            'symbol': self.symbol
        }

def get_different_period(old_candles, new_period):
    ''' convert list of candles to a new period '''
    if len(old_candles) == 0:
        return []
    old_period = old_candles[0].period_len
    if new_period % old_period != 0:
        raise ValueError('new period ' + str(new_period) + ' is not divisible by old period ' + str(old_period))
    candles_at_a_time = new_period // old_period
    new_candles = []
    for current_index in range(0, len(old_candles), candles_at_a_time):
        first_candle = old_candles[current_index]
        if current_index + candles_at_a_time - 1 >= len(old_candles):
            last_candle = old_candles[-1]
        else:
            last_candle = old_candles[current_index + candles_at_a_time - 1]
        current_candle = Candle(first_candle.symbol, first_candle.timestamp,first_candle.first, last_candle.last, period_len = new_period)
        current_volume_base = first_candle.volume_base
        current_volume_quote = first_candle.volume_quote
        low = first_candle.low
        high = first_candle.high
        for offset in range(1, candles_at_a_time):
            if current_index + offset >= len(old_candles):
                break
            this_candle = old_candles[current_index + offset]
            current_volume_base += this_candle.volume_base
            current_volume_quote += this_candle.volume_quote
            if this_candle.low < low:
                low = this_candle.low
            if this_candle.high > high:
                high = this_candle.high
        current_candle.volume_base = current_volume_base
        current_candle.volume_quote = current_volume_quote
        current_candle.low = low
        current_candle.high = high
        new_candles.append(current_candle)
    return new_candles

class Order:
    def __init__(self, symbol, timestamp=None, order_id=None, price=0.0, volume=0.0, executed_volume=0.0, side='buy', completed=False):
        self.symbol = symbol
        self.timestamp = date_to_timestamp(timestamp)
        self.order_id = order_id
        self.price = float(price)
        self.volume = float(volume)
        self.side = side.lower()
        self.completed = completed
        self.executed_volume = float(executed_volume)

    def copy(self):
        return Order(self.symbol, timestamp = self.timestamp, order_id=self.order_id, price=self.price, volume=self.volume, executed_volume=self.executed_volume, side=self.side, completed=self.completed)

    def __repr__(self):
        return 'Order (symbol: ' + self.symbol + ' time: ' + str(self.timestamp) + ' id: ' + str(self.order_id) + ' side: ' + self.side + ' price: ' + str(self.price) + ' volume: ' + str(self.volume) + ' executed: ' + str(self.executed_volume) + ')'

class Orderbook:
    def __init__(self, symbol, buy_orders=[], sell_orders=[]):
        self.symbol = symbol
        self.buy_orders = buy_orders
        self.sell_orders = sell_orders

class Exchange:
    def __init__(self, exchange_name, rest_url, auth = None, maker_fee = 0.0025, taker_fee = 0.0025, rest_requests = 10, socket_requests = 100, socket_url=""):
        super().__init__()
        self.running = True
        self.auth = auth
        self.url = rest_url
        self.socket_url = socket_url
        self.rest_session = requests.session()
        self.pending_requests = {}
        self.exchange_name = exchange_name

        # Maker and taker fees for trades (highest)
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        # Number of allowed REST requests per second
        self.rest_requests = rest_requests
        self.rest_timeout_sec = 1

        # Number of allowed websocket requests per second
        self.socket_requests = socket_requests
        self.socket_timeout_sec = 1

        # Retries before giving up
        self.retry_count = 0
        self.max_retries = 5

        # Track response statistics
        self.rest_request_count = 0
        self.avg_rest_ping_time = 0

        self.socket_request_count = 0
        self.avg_socket_ping_time = 0

        # Initialize log
        self.logger = logging.getLogger('log')

        self.socket_session = None
        self.add_callbacks()

    def __repr__(self):
        return 'Exchange (name: ' + str(self.exchange_name) + ' maker fee: ' + str(self.maker_fee) + ' taker fee: ' + str(self.taker_fee) + \
            ' url: ' + str(self.url) + ' socket url: ' + str(self.socket_url) + \
            ' retry count: ' + str(self.retry_count) + ' max retries: ' + str(self.max_retries) + \
            ' rest requests: ' + str(self.rest_request_count) + ' avg rest ping time: ' + str(self.avg_rest_ping_time) + \
            ' socket requests: ' + str(self.socket_request_count) + ' avg socket ping time: ' + str(self.avg_socket_ping_time) + ')'

    def start(self):
        self.refresh()

    def generate_id(self):
        return str(round(time.time() * 10000)) + str(uuid.uuid4())[:8]

    def print_status(self):
        print('Exchange:', self.exchange_name, "Rest Ping:", self.avg_rest_ping_time, "Socket Ping:", self.avg_socket_ping_time)

    def init_callbacks(self, symbols=[]):
        """Code to initialize callback functions if necessary for the exchange"""
        pass

    def add_callbacks(self, on_market_trade = None, on_order_executed = None, on_price_threshold = None, on_orderbook_changed = None, symbols=[]):
        """Callbacks on event occurrences"""
        self.on_market_trade = on_market_trade
        self.on_order_executed = on_order_executed
        self.on_price_threshold = on_price_threshold
        self.on_orderbook_changed = on_orderbook_changed

        self.init_callbacks(symbols=symbols)

    def add_msg(self, msg, msg_type = "generic", level=Level.DEBUG):
        """Add message to the queue of messages"""
        if level < Level.ERROR and 'error' in str(msg).lower():
            level = Level.ERROR
        if level >= Level.ERROR:
            data = {
                'timestamp':datetime.datetime.now(),
                'type':msg_type,
                'message':str(msg),
                'level':level
            }
            print(data)
        if level != Level.CRITICAL:
            self.logger.log(level * 10, 'type: ' + str(msg_type) + ' msg: ' + str(msg))
        else:
            raise Exception(str(msg))

    def get_symbol_from_name(self, name):
        """Looks up the symbol given its ID"""
        for symbol in self.symbols:
            if symbol.name == name:
                return symbol

        return None

    def get_active_orders(self):
        """Return all orders active for this account"""
        pass

    def get_rest(self, request, request_type='get', data=None, headers=None):
        """Get a rest response from the server"""

        # Get and time the response
        start = datetime.datetime.now()

        if request_type.upper() == 'GET':
            response = self.rest_session.get(self.url + request, params=data, timeout=30, headers=headers)
        elif request_type.upper() == 'PUT':
            response = self.rest_session.put(self.url + request, data=data, timeout=30, headers=headers)
        elif request_type.upper() == 'POST':
            response = self.rest_session.post(self.url + request, data=data, timeout=30, headers=headers)
        elif request_type.upper() == 'DELETE':
            response = self.rest_session.delete(self.url + request, data=data, timeout=30, headers=headers)
        else:
            self.add_msg("Invalid request type: " + request_type)
            return None

        stop = datetime.datetime.now()

        # Track the average response time
        response_time = (stop - start).microseconds
        self.rest_request_count += 1
        self.avg_rest_ping_time = (self.avg_rest_ping_time * (self.rest_request_count - 1) + response_time) / self.rest_request_count

        # Decode the response, catching any returned errors
        if response is None:
            self.add_msg('Empty response from server')
            return None
        try:
            response_json = response.json()
            display_response = str(response_json)
            if response.status_code != 200:
                self.add_msg(response.text, "network_status_code", level=Level.ERROR)
            if len(display_response) > 100:
                display_response = display_response[:100]
            self.add_msg({"Request":request, "Response": display_response}, level=Level.DEBUG)
            return response_json
        except ValueError as e:
            self.add_msg(e, level=Level.CRITICAL)
            
        return None

    def find_shortest_symbol_path(self, start_currency, stop_currency):
        symbol_queue = Queue()
        visited_currencies = []
        initial = True

        # Find shortest path using Moore's BFS algorithm
        while not symbol_queue.empty() or initial:
            if initial:
                initial = False
                path = []
                current = start_currency
            else:
                path = symbol_queue.get()
                current = path[-1]['symbol'].base if path[-1]['side'] == 'buy' else path[-1]['symbol'].quote

            for symbol in self.symbols:
                if symbol.base in visited_currencies or symbol.quote in visited_currencies:
                    continue
                if symbol.base == current or symbol.quote == current:
                    secondary = symbol.quote if symbol.base == current else symbol.base
                    side = 'buy' if symbol.quote == current else 'sell'
                    path.append({'symbol':symbol, 'side':side})
                    if secondary == stop_currency:
                        return path
                    symbol_queue.put(path.copy())
                    path.pop()
            visited_currencies.append(current)
        return []

    def authenticate(self):
        pass

    def get_orderbook(self, symbol, limit=100):
        """Get orderbook. """
        pass

    def get_increment(self, base):
        my_dict = {'BTC': 0.00001, 'ETH': 0.0001}
        if base in my_dict:
            return my_dict[base]
        else:
            raise ValueError('The increment should be found')

    def prepare_order(self, order):
        price = float(order.price)
        volume = float(order.total_amt)
        symbol = order.symbol
        if not isinstance(symbol, Symbol):
            base = symbol[:symbol.index('USD')]
            quote = symbol[symbol.index('USD'):]
            increment = self.get_increment(base)
            symbol_obj = Symbol(symbol, base, quote, base_increment=float(increment), min_value=1)
            order.symbol_obj = symbol_obj
        else:
            order.symbol_obj = symbol
        if order.side.lower() == 'buy':
            price = order.symbol_obj.string_price(order.symbol_obj.round_price(price, round_up=False))
        elif order.side.lower() == 'sell':
            price = order.symbol_obj.string_price(order.symbol_obj.round_price(price, round_up=True))
        volume = order.symbol_obj.string_volume(order.symbol_obj.round_volume(volume, round_up=False))
        return price, volume

    def transfer(self, currency_code, amount, to_exchange):
        pass

    def new_order(self, order):
        """Place an order."""
        pass

    def market_order(self, order):
        """Place a market order"""
        pass

    def replace_order(self, order):
        """Replace a previous order with new parameters"""
        pass

    def get_order(self, client_order_id, wait=None):
        """Get order info."""
        pass

    def get_balance(self, currencies=None):
        """Get balance of given currencies. If no currency given, return all nonzero balances
            Format: {'BTCUSD':{'available':0.05,'reserved':0.05,'total':0.1}}"""
        pass

    def get_trades(self, symbol, start_time = None, stop_time = None):
        """Trade history."""
        pass

    def cancel_order(self, client_order_id):
        """Cancel order."""
        pass

    def cancel_orders(self, buy_orders=True, sell_orders=True):
        """Cancel all orders filtered by buy or sell orders"""
        pass

    def get_candles(self, symbol, num_candles=20, period='5M'):
        """Get candles for the given symbol at the given increment"""
        pass

    def convert_currency_market(self, origin_currency, destination_currency, origin_volume):
        # Converts one currency to another using market buys/sells
        current_volume = origin_volume
        new_volume = 0
        path = self.find_shortest_symbol_path(origin_currency, destination_currency)
        if path:
            for trade in path:
                book = self.get_orderbook(trade['symbol'], limit=500)
                orders = book.sell_orders if trade['side'] == 'buy' else book.buy_orders
                if len(orders) == 0:
                    current_volume = 0
                    break
                i = 0
                while current_volume > 0:
                    order = orders[i]

                    if trade['side'] == 'buy':
                        volume = min(current_volume, order.price * order.volume)
                        new_volume += trade['symbol'].buyable(order.price, volume, self.taker_fee)
                    else:
                        volume = min(current_volume, order.volume)
                        new_volume += trade['symbol'].quote_if_sold(order.price, volume, self.taker_fee)
                    current_volume -= volume
                    i += 1
                current_volume = new_volume
        else:
            current_volume = 0
        return current_volume

    def get_total_value(self, to_currency="USD", origin_currencies=None, include='total', balances=None, all_funds=False):
        """Calculate total value of all assets converted into provided currency if traded immediately"""

        total = {'total':0}
        if balances is None:
            balances = self.get_balance(origin_currencies, all_funds=all_funds)
            print('BALANCES:', balances)
        for currency, balance in balances.items():
            current_volume = balance[include]
            if currency == to_currency:
                total[currency] = {'available':balance['available'],'reserved':balance['reserved'],'total':current_volume}
                total['total'] += current_volume
                continue
            proportion_available = balance['available'] / balance['total'] if balance['total'] > 0 else 1
            current_volume = self.convert_currency_market(currency, to_currency, current_volume)

            total['total'] += current_volume
            total[currency] = {'available':current_volume * proportion_available,
                               'reserved':current_volume * (1 - proportion_available),
                               'total':current_volume}
        self.add_msg('Total Value: ' + str(total['total']), Level.DEBUG)
        return total

    def print_statistics(self):
        print("REST average ping time:", avg_rest_ping_time, "\nSocket average ping time:", avg_socket_ping_time)

    def refresh(self):
        """Refresh currency info"""
        self.retry_count = 0

        # Initialize generic exchange dictionary
        self.currencies = []
        self.symbols = []

    def exit(self):
        """Perform necessary steps to exit the program"""
        self.running = False
