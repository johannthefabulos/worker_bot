import uuid
import time
import datetime

import hmac
import hashlib

import requests
from decimal import *
from exchanges.Exchange import *
import threading
from exchanges.socket import *

class hitbtc(Exchange):
    def __init__(self, public_key=None, secret=None):
        auth = None
        if public_key and secret:
            auth = True
        super().__init__('HITBTC', 'https://api.hitbtc.com/api/2', auth=auth, rest_requests = 100, socket_requests = 1000000, socket_url='wss://api.hitbtc.com/api/2/ws')
        self.public_key = public_key
        self.secret = secret
        
        self.rest_session = requests.session()
        
        self.keep_running = True
        self.socket_session = WebSocketConnectorThread(self.socket_url, self.on_socket_message, self.on_socket_close, self.on_socket_error, self.on_open)
        self.socket_session.start()
        
        self.run_event = threading.Event()
        self.run_event.wait(timeout=10)
        self.run_event = None
        
    def authenticate(self, public_key, secret):
        self.rest_session.auth = (public_key, secret)
        
        if self.socket_session and self.socket_session.is_connected():
            nonce = str(round(time.time() * 1000))
            signature = hmac.new(secret.encode('UTF-8'), nonce.encode('UTF-8'), hashlib.sha256).hexdigest()
            request = {'method':'login','params':{'algo':'HS256','pKey':public_key,'nonce':nonce,'signature':signature}}
            response = self.socket_send(request)

            if 'error' in str(response):
                self.add_msg('Login failed: ' + response['error'], 'socket_login_error')

    def get_orderbook(self, symbol, limit=100):
        """Get orderbook. """
        data = {'limit':limit}
        book = self.get_rest("/public/orderbook/" + symbol.name, data=data)
        
        buy_orders=[]
        for buy_order in book['bid']:
            buy_orders.append(Order(symbol, price=float(buy_order['price']), volume=float(buy_order['size'])))
        sell_orders=[]
        for sell_order in book['ask']:
            sell_orders.append(Order(symbol, price=float(sell_order['price']), volume=float(sell_order['size'])))
            
        return Orderbook(symbol, buy_orders, sell_orders)
        
    def get_balance(self, currencies=None, all_funds=False):
        returned_balances = []
        response = self.get_rest("/trading/balance")
        if all_funds:
            account_balance=self.get_rest("/account/balance")
        
        tracked_currencies = {}
        for currency in response:
            if all_funds:
                currency['reserved'] = float(currency['reserved'])  + (
                    sum([float(acct_currency['available']) for acct_currency in account_balance if acct_currency['currency'] == currency['currency']]))
            if (currencies is None and (float(currency['available']) > 0 or float(currency['reserved']) > 0)) or (
                currencies is not None and currency['currency'] in currencies):
                total_balance = float(currency['available']) + float(currency['reserved'])
                tracked_currencies[currency['currency']] = {
                    'available':float(currency['available']),
                    'reserved':float(currency['reserved']),
                    'total':total_balance}
        return tracked_currencies
        
    def get_active_orders(self):
        """Return all orders active for this account"""
        orders = []
        order_data = self.get_rest('/order/')
        for order in order_data:
            symbol = self.get_symbol_from_name(order['symbol'])
            volume = float(order['quantity']) - float(order['cumQuantity'])
            orders.append(Order(symbol, timestamp=order['createdAt'], order_id=order['clientOrderId'], price=order['price'],
                            volume=volume, side=order['side'], executed_volume=float(order['cumQuantity'])))
        return orders

    def new_order(self, order, force_limit=False):
        """Place an order."""
        price, volume = self.prepare_order(order)

        if order.symbol.prices_equal(order.price, 0):
            print("Error: attempt to place order at price of 0")
            return None
        elif order.symbol.volumes_equal(order.volume, 0):
            print("Error: attempt to place order at volume of 0")
            return None
            
        if self.socket_session and self.socket_session.is_connected():
            data = {'method':'newOrder', 'params':{
                'clientOrderId':order.order_id,
                'symbol':order.symbol.name,
                'side':order.side,
                'price':price,
                'quantity':volume
            }}
            
            if order.order_id is None:
                data['params']['clientOrderId'] = self.generate_id()
            print("New", order.side, "order for", order.symbol.name, "at price of", price, "and volume of", volume, "id:", data['params']['clientOrderId'])
            response = self.socket_send(data)
            self.add_msg('Response: ' + str(response))
            if 'error' in response:
                self.add_msg('Order place error: ' + str(response['error']) + ' request: ' + str(data))
                print('Order place error: ' + str(response['error']) + ' request: ' + str(data))
                return None
            return response['result']['clientOrderId']
        else:
            print("New", order.side, "order for", order.symbol.name, "at price of", price, "and volume of", volume)
            data = {'symbol': order.symbol.name, 'side': order.side, 'quantity': volume, 'price': price}
            if order.order_id:
                data['clientOrderId'] = order.order_id
            response = self.get_rest('/order/', data=data, request_type='put')
            if 'error' in response:
                self.add_msg('Order place error: ' + str(response['error']) + ' request: ' + str(data))
                return None
            return response['clientOrderId']
            
    def market_order(self, order):
        volume = order.symbol.string_price(order.volume)
        
        if order.symbol.volumes_equal(order.volume, 0):
            print('Error: attempt to place order at volume of 0')
            return None
            
        data = {'symbol':order.symbol.name, 'side': order.side, 'quantity': volume, 'type': 'market'}
        response = self.get_rest('/order/', data=data, request_type='put')
        if 'error' in response:
            self.add_msg('Order place error: ' + str(response['error']) + ' request: ' + str(data))
            return None
        return response['clientOrderId']

    def get_order(self, order, wait=None):
        """Get order info."""
        data = {'wait': wait} if wait is not None else {}

        return self.get_rest("/order/" + str(order.order_id), data=data)

    def cancel_order(self, order):
        """Cancel order."""
        print("Cancel order", order)
        if self.socket_session and self.socket_session.is_connected():
            data = {'method':'cancelOrder','params':{
                'clientOrderId':order.order_id
            }}
            self.socket_send(data)
        else:
            self.get_rest("/order/" + str(order.order_id), request_type="delete")
        
    def cancel_orders(self, buy_orders=True, sell_orders=True):
        """Cancel all orders filtered by buy or sell orders"""
        if buy_orders and sell_orders:
            self.get_rest("/order/", request_type="delete")
        else:
            orders = self.get_active_orders()
            for order in orders:
                if sell_orders and order.side == 'sell':
                    self.cancel_order(order)
                elif buy_orders and order.side == 'buy':
                    self.cancel_order(order)
        
    def get_candles(self, symbol, num_candles=20, period=5):
        """Get candles for the given symbol at the given increment"""
        periods = {1:'M1',3:'M3',5:'M5',15:'M15',30:'M30',60:'H1',240:'H4',1440:'D1',10080:'D7',43200:'1M'}

        # We must also fill in candles from periods in which no activity occurred
        factor = 60 * period
        start = datetime.datetime.utcfromtimestamp(time.time() // factor * factor) - datetime.timedelta(minutes=period * num_candles)
        candles = []
        
        for offset in range(100):
            start_str = start.strftime("%Y-%m-%dT%H:%M:%S.000")
            response = self.get_rest('/public/candles/' + symbol.name, data={'period':periods[period],'limit':1000,'offset':offset * 1000,'from':start_str})
            for candle in response:
                candles.append(Candle(symbol, candle['timestamp'], first=candle['open'], last=candle['close'], 
                                low=candle['min'], high=candle['max'], volume_base=candle['volume'], volume_quote=candle['volumeQuote'], period_len=period))
            if len(response) < 1000:
                break
            
        candles.append(Candle(symbol, datetime.datetime.now()))
        new_candles = []
        for i in range(1, len(candles)):
            candle = candles[i - 1]
            next_candle = candles[i]
            new_candles.append(candle.copy())
            while (next_candle.timestamp.replace(tzinfo=None) - candle.timestamp.replace(tzinfo=None)).total_seconds() > period*60:
                candle.timestamp += datetime.timedelta(seconds=period*60)
                candle_copy = Candle(candle.symbol, candle.timestamp, first=candle.last, last=candle.last, high=candle.last, low=candle.last)
                new_candles.append(candle_copy)
        
        return new_candles[-1 * num_candles:]
        
    def replace_order(self, order, new_id=None):
        """Replace a previous order with new parameters"""
        if order.symbol.prices_equal(order.price, 0):
            print("Error: attempt to place order at price of 0")
            return None
        elif order.symbol.volumes_equal(order.volume, 0):
            print("Error: attempt to place order at volume of 0")
            return None
        if self.socket_session and self.socket_session.is_connected():
            data = {'method':'cancelReplaceOrder', 'params':{
                'clientOrderId':order.order_id,
                'quantity':order.symbol.string_volume(order.volume + order.executed_volume),
                'price':order.symbol.string_price(order.price)
            }}
            
            if new_id:
                data['params']['requestClientId'] = new_id
            else:
                data['params']['requestClientId'] = self.generate_id()
            
            print('Replacing', order.symbol.name, 'id:', order.order_id, 'with new', order.side, 'order at price of', order.symbol.string_price(order.price), 'and volume of', order.symbol.string_volume(order.volume), 'new id:', data['params']['requestClientId'])
            response = self.socket_send(data)
            if 'error' in response:
                self.add_msg('Order place error: ' + str(response['error']) + ' request: ' + str(data))
                return None
            return response['result']['clientOrderId']
        else:
            print('Replacing', order.symbol.name, 'id:', order.order_id, 'with new', order.side, 'order at price of', order.symbol.string_price(order.price), 'and volume of', order.symbol.string_volume(order.volume))
            self.cancel_order(order)
            if new_id:
                order.clientOrderId = new_id
            else:
                order.clientOrderId = None
            return self.new_order(order)
            
    def get_account_trades(self, number=100, symbol=None):
        """Get recent trades for this account"""
        data = {'limit':number}
        if symbol:
            data['symbol'] = symbol.name
        response = self.get_rest("/history/trades", data=data)
        trades = []
        for trade in response:
            trades.append(Trade(self.get_symbol_from_name(trade['symbol']), trade['timestamp'], price=trade['price'], volume=trade['quantity'], fee=trade['fee'], side=trade['side']))
            
        return trades

    def get_transaction(self, transaction_id):
        """Get transaction info."""
        return self.get_rest("/account/transactions/" + transaction_id)

    def get_ticker(self, name):
        return self.get_rest("/public/ticker/" + name)
        
    def on_open(self):
        #self.add_msg("Socket started", "socket_msg")
        threading.Thread(target=self.refresh).start()
        
    def on_socket_close(self):
        self.add_msg("Socket closed", "socket_msg")
        pass
            
    def on_socket_error(self, error):
        self.add_msg(str(error), 'socket_error')
            
    def socket_send(self, data, timeout=10, callback=None):
        while self.retry_count < self.max_retries and self.socket_session:
            # Generate a unique ID
            request_id = int(10000 * time.time())
            while request_id in self.pending_requests:
                request_id = int(10000 * time.time())
                
            data['id'] = request_id
            
            # Start timer
            start = datetime.datetime.now()
            
            if timeout > 0:
                wait_event = threading.Event()
                
                self.pending_requests[request_id] = {
                    'request':data,
                    'start':start,
                    'stop':None,
                    'event':wait_event,
                    'callback':callback,
                    'response':None
                }
                
                self.socket_session.send(data)
                
                # Wait for response
                self.pending_requests[request_id]['event'].wait(timeout=timeout)
                
                # Return response
                response = self.pending_requests[request_id]['response']
                if callback is None:
                    self.pending_requests.pop(request_id)
                else:
                    self.pending_requests[request_id]['event'] = None
                
                return response
            
            else:
                if callback:
                    self.db_file_name[request_id] = {
                        'request':data,
                        'start':start,
                        'stop':None,
                        'event':None,
                        'callback':callback,
                        'response':None
                    }
                
                self.socket_session.send(data)
        return None
        
    def on_socket_message(self, data, time):
        received_time = datetime.datetime.now()
        
        if 'id' in data and data['id'] in self.pending_requests:
            request = self.pending_requests[data['id']]
            
            if request['stop'] is None:
                request['stop'] = received_time
                response_time = (received_time - request['start']).microseconds
                self.socket_request_count += 1
                self.avg_socket_ping_time = (self.avg_socket_ping_time * (self.socket_request_count - 1) + response_time) / self.socket_request_count

            if request['event']:
                request['response'] = data
                self.pending_requests[data['id']]['event'].set()
                
            if request['callback']:
                request['callback'](data)
                
        else:
            if 'method' in data:
                if data['method'] == 'report' and data['params']['reportType'] == 'trade':
                    self.order_executed_handler(data)
                elif data['method'] == 'snapshotOrderbook' or data['method'] == 'updateOrderbook':
                    self.orderbook_update_handler(data)
                elif data['method'] == 'updateTrades':
                    self.market_trade_handler(data)
            else:
                self.add_msg('message id not in requests dictionary. Message: ' + str(data), 'socket_receive_error')
        
    def order_executed_handler(self, data):
        data = data['params']
        if data['status'] != 'new' and data['type'] == 'limit':
            symbol = self.get_symbol_from_name(data['symbol'])
            order = Order(symbol, timestamp=data['createdAt'],
                         order_id=data['clientOrderId'], price=data['tradePrice'], volume=data['tradeQuantity'],
                         side=data['side'], completed=True if data['status'] == 'filled' else False, executed_volume=data['cumQuantity'])
            trade = Trade(symbol, datetime.datetime.now(), price=data['tradePrice'], volume=data['tradeQuantity'],
                         side=data['side'], fee = data['tradeFee'])
            threading.Thread(target=self.on_order_executed, args=(order,trade,)).start()
            
    def orderbook_update_handler(self, data):
        data = data['params']
        symbol = self.get_symbol_from_name(data['symbol'])
        buy_orders = [Order(symbol, data['timestamp'], price=float(order['price']),volume=order['size'],side='buy') for order in data['bid']]
        sell_orders = [Order(symbol, data['timestamp'], price=float(order['price']),volume=order['size'],side='sell') for order in data['ask']]
        orderbook = Orderbook(symbol, buy_orders=buy_orders, sell_orders=sell_orders)
        print("ORDERBOOK_UPDATE")
        threading.Thread(target=self.on_orderbook_changed, args=(orderbook,data['sequence'])).start()
    
    def market_trade_handler(self, data):
        data = data['params']
        symbol = self.get_symbol_from_name(data['symbol'])
        trades = [Trade(symbol, trade['timestamp'], price=float(trade['price']), volume=float(trade['quantity']), side=trade['side']) for trade in data['data']]
        print("TRADE UPDATE")
        threading.Thread(target=self.on_market_trade, args=(trades,)).start()      
        
    def init_callbacks(self, symbols=[]):
        """Code to initialize callback functions if necessary"""
        if self.on_order_executed:
            self.socket_send({'method':'subscribeReports','params':{}})
        for symbol in symbols:
            if self.on_orderbook_changed:
                self.socket_send({'method':'subscribeOrderbook', 'params':{'symbol':symbol.name}})
            if self.on_market_trade:
                self.socket_send({'method':'subscribeTrades', 'params':{'symbol':symbol.name}})
        
    def refresh(self):
        super().refresh()
        
        self.rest_session = requests.session()
        
        if self.auth:
            self.authenticate(self.public_key, self.secret)
        
        # Populate the currencies
        currencies = self.get_rest("/public/currency")
        self.currencies = []
        for currency in currencies:
            self.currencies.append(currency['id'])
        
        # Populate the symbols
        first = True
        symbols = self.get_rest("/public/symbol")
        self.symbols = []
        for symbol in symbols:
            if first:
                # Get the maker and taker fees
                first = False
                self.maker_fee = float(symbol['provideLiquidityRate'])
                self.taker_fee = float(symbol['takeLiquidityRate'])
                
            temp_symbol = Symbol(
                name=symbol['id'],
                base=symbol['baseCurrency'],
                quote=symbol['quoteCurrency'],
                base_increment=float(symbol['quantityIncrement']),
                quote_increment=float(symbol['tickSize']))
                
            self.symbols.append(temp_symbol)
        self.init_callbacks()
        
        if self.run_event:
            self.run_event.set()
        
    def exit(self):
        """Perform necessary steps to exit the program"""
        print("Exiting...")
        if self.socket_session:
            self.socket_session.disconnect()
        self.running = False