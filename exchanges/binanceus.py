from exchanges.binance import binance


class binanceus(binance):
    def __init__(self, public_key=None, secret=None, auth=True, trigger_dict=None, queue_dict=None):
        super().__init__(public_key=public_key, secret=secret, auth=auth, endpoint='https://api.binance.us', trigger_dict=trigger_dict, queue_dict=queue_dict, ws_endpoint='wss://stream.binance.us:9443', name_override='BINANCEUS')

