import numpy as np

def scaler(val, mi, ma, lo=0., hi=1.):
    try:
        return (val-mi)/(ma-mi)*(hi-lo) + lo
    except TypeError:
        return -1.

def compute_month(dts, npad):
    return {
        'month':list(map(lambda x: x.month, dts))+[0]*npad, 
#         'month_mask':[1]*len(dts)+[0]*npad,
    }

def compute_day(dts, npad):
    return {
        'day':list(map(lambda x: x.day, dts))+[0]*npad, 
#         'day_mask':[1]*len(dts)+[0]*npad,
    }

def compute_wday(dts, npad):
    return {
        'wday':list(map(lambda x: x.weekday()+1, dts))+[0]*npad, 
#         'wday_mask':[1]*len(dts)+[0]*npad,
    }

def _wtime(dt):
    seconds, day = (dt.hour*60 + dt.minute)*60 + dt.second, 24*60*60
    time_sin, time_cos = np.sin(2*np.pi*seconds/day), np.cos(2*np.pi*seconds/day)
    return time_sin, time_cos

def compute_wtime(dts, npad):
    cyclic_times = list(map(_wtime, dts))
    return {
        'oddtime':[x[0] for x in cyclic_times]+[0]*npad, 
        'eventime':[x[1] for x in cyclic_times]+[0]*npad,
    }

def compute_srel(dts, npad, dt_min, srel_min, srel_max):
    return {
        'srel':list(map(lambda x: scaler((x-dt_min).total_seconds(), srel_min, srel_max), dts))+[0]*npad, 
#         'srel_mask':[1]*len(dts)+[0]*npad,
    }