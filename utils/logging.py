'''
    Functions to log translator usage
'''
import pytz
import csv
from datetime import datetime
from os import path

def log_usage(src_txt, trg_txt,feedback=None,model_v=None,inf_time=None):
    log_fn='usage_logs2.csv'
    log_dir='logs'
    feedback='na'
    naive_dt = datetime.now()
    tz='Europe/Dublin'
    indy = pytz.timezone(tz)
    dt = indy.localize(naive_dt)

    fields=[dt,tz,src_txt,trg_txt,feedback,inf_time]

    if path.exists(log_fn):
        with open(log_dir + '/' + log_fn, 'a') as f:
            writer = csv.writer(f,delimiter=',')
            writer.writerow(fields)
    else:
         with open(log_dir + '/' + log_fn, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(fields)