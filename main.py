import asyncio
import copy
import logging
import multiprocessing
import os
import signal
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from queue import Queue

import pandas as pd

from account import Account
from exchanges import Exchange, binanceus
from exchanges.Exchange import Candle, Order, Trade
from exchanges.fake_exchange import FakeExchange
from manage.managetrades import manage_trades, suggested_trade
from misc import crypto
from tests.downloader import downloader
from tests.limit_strats import avg_strat_tiers, generic_limit_strat, test_strat
from tests.market_strats import market_strat_support, similarity_test
from tests.Strats import Permutation


class setup:
    def __init__(self, symbol, myDict, simulating=False, start_time=None, end_time=None, obj_to_use=None, username=None, request=None, trigger_event=None, trigger_queue = None):
        self.SYMBOL = symbol
        self.trigger_event = trigger_event
        self.trigger_queue = trigger_queue
        self.simulating = simulating
        self.used_account = None
        self.used_exchange = None
        self.order_executed_obj = None
        self.myDict = myDict
        self.start_time = start_time
        self.end_time = end_time
        self.obj_to_use = obj_to_use
        self.username = username
        self.request = request
        self.all_managers = []

    def setup_exchanges_to_use(self):
        if self.simulating:
            self.used_account = Account(dict_of_amounts=self.myDict, simulating=self.simulating)
            sim_exchange = FakeExchange(self.used_account, self.start_time, self.end_time, trigger_event=self.trigger_event, trigger_queue=self.trigger_queue)
            self.used_exchange = sim_exchange
            self.used_exchange.set_dict(self.myDict)
        else:
            # TODO: figure out trigger event and queue for real exchange
            api_info = crypto.get_api_info()
            private = api_info['private']
            public = api_info['public']
            self.used_exchange = binanceus(public_key=public, secret=private, trigger_dict=self.trigger_event, queue_dict=self.trigger_queue)
            # self.used_exchange.start()
            self.used_account = Account(dict_of_amounts=myDict, exchange=self.used_exchange, simulating=self.simulating)
        return self.used_exchange, self.used_account

    def get_strat_ready(self):
        i = 0
        if self.simulating:
            start_time = self.start_time
        else:
            start_time = datetime.utcnow()
        # if self.obj_to_use is not None:
        #     used_strat1 = self.obj_to_use(exchange=self.used_exchange, account=self.used_account, manager_id=0, fucn=lambda x: x ** 2, symbol=self.SYMBOL, percent_to_manage=.03, start_time_iso=start_time)
        # else:
        percent_manage = .5
        copy_percent_to_manage = copy.deepcopy(percent_manage)
        copy_symbol = copy.deepcopy(self.SYMBOL)
        used_strat1 = test_strat(exchange=self.used_exchange, account=self.used_account, manager_id=0, symbol=copy_symbol, percent_to_manage=copy_percent_to_manage)
        lookback = None
        average_periods = None
        new_exchange = None
        strat_copy = None
        perms1 = Permutation(lookback, average_periods, **used_strat1.params)
        all_managers = []
        while not perms1.is_done():
            lookback, average_periods, used_strat1.params = perms1.next_perm()
            new_strat = copy.copy(used_strat1)
            if self.simulating:
                new_exchange = FakeExchange(self.used_account, self.start_time, self.end_time, trigger_event=self.trigger_event, trigger_queue=self.trigger_queue, trigger_id = i)
                new_exchange.set_dict(self.myDict)
                # my_new_exchange = FakeExchange(self.used_account, self.start_time, self.end_time)
                new_strat.exchange = new_exchange
            copy_symbol = copy.deepcopy(self.SYMBOL)
            strat_copy = copy.copy(new_strat)
            copy_percent = copy.copy(percent_manage)
            manager = manage_trades(strat_copy, strat_copy.exchange, i, copy_symbol, copy_percent)
            strat_copy.manager_id = manager
            all_managers.append(manager)
            # tryer = None
            # unpicklable = []
            # new_exchange
            # for key, value in manager.strat.exchange.server_obj.__dict__.items():

            #     try:
            #         pickle.dumps(value)
            #         tryer = True
            #     except Exception as e:
            #         print(f"error pickling object {e}")
            #         tryer = False
            #     if not tryer:
            #         unpicklable.append((key,value))
            # can't run this yet because we need to run it after fake_server is populated with downloader(run_once())
            # used_strat1.run_once()
            # manager.append_amt_objects(used_strat1.get_all_amt_objects())
            i += 1
        put_dict = {}
        put_dict['num_graphs'] = len(all_managers)
        graphs_queue.put(put_dict)
        trigger_num_graphs.set()
        self.all_managers=all_managers
        beg_managers = all_managers
        # if not self.simulating:
        #     self.order_executed_obj = on_order_executed(self.all_managers)
            # self.used_exchange.add_callbacks(on_order_executed=self.order_executed_obj.on_trade_fulfilled)
        # self.all_managers = self.used_account.get_all_managers()


class run:
    def __init__(self, setup):
        _, _ = setup.setup_exchanges_to_use()
        setup.get_strat_ready()
        self.trigger_event = setup.trigger_event
        self.trigger_queue = setup.trigger_queue
        self.start_time = setup.start_time
        self.end_time = setup.end_time
        self.used_account = setup.used_account
        self.used_exchange = setup.used_exchange
        self.all_managers = setup.all_managers
        self.simulating = setup.simulating
        self.start_time = setup.start_time
        self.end_time = setup.end_time
        self.SYMBOL = setup.SYMBOL
        self.increment = None
        self.obj_to_use = setup.obj_to_use
        self.username = setup.username
        self.request = setup.request
        self.all_needed_dfs = None
        self.furthest_back_time = 90
        self.run_once = False

    def run_downloader(self, manager):
        period_size = []
        all_needed_dfs = None
        my_downloader = downloader(start_time=self.start_time, end_time=self.end_time, symbol=self.SYMBOL)
        print(manager.strat.params['furthest_back'])
        self.furthest_back_time = manager.strat.params['furthest_back'] * manager.strat.params['lookback'] * 60
        # This function downloads info from an exchange(binance right now) and saves them into {self.SYMBOL}.csv (get_indicies)
        success, used_start = my_downloader.get_indicies(furthest_back_time_seconds=self.furthest_back_time, manager=manager)
        if not success:
            raise ValueError("There was a problem with the downloader")
        self.all_needed_dfs=my_downloader.get_dfs_from_file(manager.strat.params['lookback'], used_start)
        hello = 'hello'
        # This looks important I'm gonna keep it here for when we do bugs later
        # master_byte_elements = []
        # candles_keys = candles_dict.keys()
        # for key in candles_keys:
        #     candles = candles_dict[key]
        #     bytes_list = bytes_dict[key]
        #     bytes_list.pop(0)
        #     for period_len_bytes in bytes_list:
        #         byte_string = ''
        #         for byte_element in period_len_bytes:
        #             byte_string += str(byte_element) + ' '
        #         copy_string = copy.copy(byte_string)
        #         master_byte_elements.append(copy_string)    
        # all_dfs = my_downloader.create_files(bytes_dict, candles_dict, self.increment)
        # my_downloader.create_csv(all_dfs, self.SYMBOL)
        # TODO: was need to send all_needed_dfs to strat and run_once before we get too far
        print('done downloading all the stuff')
        return self.all_needed_dfs

    def create_fig(self, times, btc_prices, bot_funds):
        fig = go.Figure()
        # fig.add_trace(go.Scatter(x = times, y = bot_funds, name = 'bot_funds',mode = 'lines+markers'))
        fig.add_trace(go.Scatter(x = times, y = btc_prices, name = 'btc_prices', mode = 'lines+markers'))
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x = times, y = bot_funds, name = 'bot_funds',mode = 'lines+markers'))
        return fig, fig1

    def run_sim(self, manager):
        threads = []
        btc_prices = []
        bot_funds = []
        times = []
        sim_increment_time_minutes = manager.strat.params['lookback']
        self.increment = sim_increment_time_minutes
        candles_dict = self.run_downloader(manager)
        # quarter_points = [.25,.5,.75,1]
        candles_df = candles_dict[self.increment]
        # start_location = int(self.furthest_back_time/self.incement_time_minutes)
        manager.exchange.set_sim_dfs(self.start_time, candles_df, candles_dict)
            # TODO
            # Saw someting funky happen with this before while analyzing run_once
        manager.strat.run_once()
        manager.append_amt_objects(manager.strat.list_of_objs)
        manager.strat.exchange.set_manager(manager)
        manager.strat.exchange.start()

    def run_once_startup(self, manager):
        manager.strat.run_once()
        manager.append_amt_objects(manager.strat.list_of_objs)
        
    def handle_trigger_period(self):
        if self.trigger_event is not None:
            trigger_dict = {}
            trigger_dict['value'] = self.used_exchange.get_current_price(self.SYMBOL)
            trigger_dict['name'] = datetime.utcnow().isoformat()
            trigger_dict['trigger_id'] = 0
            self.trigger_queue['period'].put(trigger_dict)
            self.trigger_event['period'].set()

    def run_live(self, manager):
        self.handle_trigger_period()
        manager.strat.decision()
        manager.exchange.check_undone_objects(None, None, None, None, None)

    def start(self):
        all_processes = []
        if not self.simulating:
            for manager in self.all_managers:
                if not self.run_once:
                    self.run_once_startup(manager)
                    self.run_once = True
                # Run at the start of the next period
                INTERVAL_SECONDS = manager.strat.params['lookback']*60
                conditions = manager.strat.params['activate']
                interval_minute = (INTERVAL_SECONDS // 60)
                now = datetime.utcnow()
                rounded_minute = now.minute // interval_minute * interval_minute
                call_time = now.replace(minute=rounded_minute, second=conditions, microsecond=0) + timedelta(minutes=interval_minute)
                call_in_seconds = (call_time - now).total_seconds() + 5
                self.run_live(manager=manager)
                if call_in_seconds < 30:
                    call_in_seconds += INTERVAL_SECONDS
                timer = threading.Timer(call_in_seconds, self.start)
                timer.start()
        else:
            for idx, manager in enumerate(self.all_managers):
                print('manager', manager)
                # self.run_sim(manager)
                all_processes.append(multiprocessing.Process(target=self.run_sim, args=(manager,)))
            hello = 'hello'
            for idx, process in enumerate(all_processes):
                process.start()
            for process in all_processes:
                process.join()
            # Only called when not running in simulation mode
            # data = get_test_data(100)
            # self.on_period_update(data.iloc[-1])

def setup_file(path, raw_code):
    my_file = open(path, "w")
        
    my_file.write(raw_code)
    # python_interpreter = "C:\\Users\\johan\\AppData\\Local\\Microsoft\\WindowsApps\\python"
        # Run the code using subprocess
    # process = subprocess.Popen(args=["python", path ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # my_file.close()
    # Wait for the process to finish and get the output
    # stdout, stderr = process.communicate()

    # Print the output
    # print("Standard Output:")
    # print(stdout)

    # print("\nStandard Error:")
    # print(stderr)
    my_file.close()
    return 'written'


class dates:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.simulating = None

    def set_start(self, start):
        self.start = start

    def set_end(self, end):
        self.end = end

    def set_sim(self, sim):
        self.simulating=sim

    def get_sim(self):
        return self.simulating

    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end


class define_obj:
    def __init__(self, obj, username, request):
        self.obj = obj
        self.username = username
        self.request = request

    def get_obj(self):
        return self.obj

    def get_username(self):
        return self.username
    
    def get_request(self):
        return self.request

    def set_obj(self, obj, username, request):
        self.obj = obj
        self.username = username
        self.request = request

    def set_username(self, username):
        self.username = username
    

# app1 = Flask(__name__)
# app1.config['SECRET_KEY'] = 'secret!'

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# SESSION_TYPE = 'filesystem'
# app.config.from_object(__name__)
# Session(app)
# app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)
# CORS(app1, resources={r"/*": {"origins": "*"}})
# CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})
# async_mode used to be 'threading' if something doesn't work out with asyncio then change it back to threading




# socketio1 = SocketIO(app1, cors_allowed_origins="*")
event_queue = multiprocessing.Queue()
period_queue = multiprocessing.Queue()
trade_queue = multiprocessing.Queue()
graphs_queue = multiprocessing.Queue()
live_trades_queue = multiprocessing.Queue()

live_period_lock = asyncio.Lock()
live_period_queue = Queue()
trigger_live_period = asyncio.Event()

live_pending_queue = multiprocessing.Queue()
trigger_live_trades = multiprocessing.Event()
trigger_live_pending = multiprocessing.Event()
trigger_event = multiprocessing.Event()
trigger_period_update = multiprocessing.Event()
trigger_trade = multiprocessing.Event()
trigger_num_graphs = multiprocessing.Event()
my_lock = multiprocessing.Lock()
another_lock = multiprocessing.Lock()
graph_lock = multiprocessing.Lock()
live_pending_lock = multiprocessing.Lock()
live_trade_lock = multiprocessing.Lock()
user_session_ids = {}
pause_event = threading.Event()
pause_event.set()
start_time = None
end_time = None
beg_managers = []
client_tasks = defaultdict()
# def cpu_bound(runner):
#     runner.start()

# logging.basicConfig(level=logging.DEBUG)


def shutdown(signal, frame):
    print(f"Received signal {signal}. Shutting down gracefully...")
    # Perform any cleanup here (e.g., stop background threads)
    exit(0)

def get_space_backwards(text, start):
    for i in range(start, -1, -1):
        if text[i] == " ":
            return text[i+1:start]
    return -1

def check_malware(text):

if __name__ == '__main__':
    # while task_data == None:
    #     continue
    # print('task_data: ', task_data)
    signal.signal(signal.SIGINT, shutdown)  # Handles Ctrl+C
    signal.signal(signal.SIGTERM, shutdown)
    simulating = True
    # for any tasks for all clients
    # loop = asyncio.get_event_loop()
    # loop.create_task(background_task())
    
    # user_db = database('user_db')
    # users = user_db.get_collection('users')
    # user_preferences = user_db.get_collection('user_preferences')
    # passkey = False
    
    fake_currency = os.getenv('fake_currency')
    # ex: fake_currency = {'BTC': 0, 'USD': 10000}
    SYMBOL = os.getenv('fake_symbol')
    # ex: SYMBOL = 'BTCUSDT'
    DATES = os.getenv('fake_dates')
    # ex: DATES = ['2023-11-03T11:00:00.000000+00:00', '2023-12-03T00:00:00.000000+00:00'] 
    DATE_OBJ = dates(DATES[0], DATES[1])
    # ex see dates class
    STRATEGY = os.getenv('strategy')
    # TODO: put in separate file
    path = r"worker_bot/tests/web_limit_strats.py"
    result = setup_file(path=path, raw_code=STRATEGY)
    # TODO: get name of class
    strat_name = get_space_backwards(STRATEGY, STRATEGY.find('(generic_limit_strat)'))
    if start_time == -1:
        print('error')
    # TODO: run anti malaware
    # TODO: get anem of class to where it needs to go(inside setup)
    web_class = getattr(path, strat_name)
    # TODOsss
    # print('last', passkey)
    # date_obj = dates(None, None)
    
    # print('succeed: ', start_time, end_time)

    keep_running = True
    myobj = define_obj(None, None, None)
    if DATE_OBJ.get_sim():
        trigger_event_dict = {'period': trigger_period_update, 'trade': trigger_trade}
        trigger_queue_dict = {'period': period_queue, 'trade': trade_queue}
    else:
        trigger_event_dict = {'period': trigger_live_period, 'trade': trigger_live_trades, 'pending': trigger_live_pending}
        trigger_queue_dict = {'period': live_period_queue, 'trade': live_trades_queue, 'pending': live_pending_queue}

    setup_exchanges = setup(symbol=SYMBOL, myDict=fake_currency, simulating=True, start_time=DATE_OBJ.get_start(), end_time=DATE_OBJ.get_end(), username=myobj.get_username(), request=myobj.get_request(), trigger_event=trigger_event_dict, trigger_queue=trigger_queue_dict)
    runner = run(setup_exchanges)
    runner.start()