from timestamp_conversions import timestamp_conversions as tconv
from Abyiss_py_requests import Abyiss_py_requests

start = 1514782800
end = 	1546318799

print(f'Start: {tconv.get_unix_to_readable(start)}, End: {tconv.get_unix_to_readable(end)}')

master = Abyiss_py_requests.AbyissRequests(host = '68.226.85.254:3000', ticker_list = ["AAPL", "AFL", "AMZN", "F"], type='stock')

master = master.iterate(start_timestamp = start, end_timestamp= end, clear_master_data=False, save_to_csv=True)