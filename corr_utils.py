import numpy as np
import pandas as pd
from math import log, exp, sqrt
from datetime import datetime, date, time, timedelta
import scipy.stats as st

def fisher_z(r):
    """
    Return the Fisher Z transformation for some empirical correlation coeff. r
    """
    r = max(min(r,1.0),-1.0)
    return log((1+r+1e-15)/(1-r+1e-15))/2

def corr_interval(r, n, alpha):
    """
    Return the (1-\alpha) confidence interval for correlation around the empirical correlation r
    Requires n>3
    """
    if n>3:
        z_r = fisher_z(r)
        z_a = st.norm.ppf(1.-(alpha/2))
        L = z_r - (z_a/sqrt(n-3))
        U = z_r + (z_a/sqrt(n-3))
        r_L = (exp(2*L)-1)/(exp(2*L)+1)
        r_U = (exp(2*U)-1)/(exp(2*U)+1)

        return r_U, r_L
    else:
        return 1, -1
    
def pearson_corr(x, y):
    """
    Calculate the correlation of the vectors x and y
    """
    n = min(len(x),len(y))
    if n==0:
        return 0.0,n
    else:
        x = x[0:n]
        y = y[0:n]
        x_bar = np.mean(x)
        y_bar = np.mean(y)
        cov = np.sum(np.multiply((x-x_bar),(y-y_bar)))
        var_x = np.sum(np.square(x-x_bar))
        var_y = np.sum(np.square(y-y_bar))
        if var_x<=0 or var_y<=0:
            return 0.0, n
        else:
            r = min(cov/(sqrt(var_x)*sqrt(var_y)),1)
            return r, n

def hour_rounder(t):
    """
    return a datetime rounded to the nearest hour
    """
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))

def minute_rounder(t):
    """
    return a datetime rounded to the nearest minute
    """
    return (t.replace(second=0, microsecond=0, minute=t.minute, hour=t.hour)
               +timedelta(minutes=t.second//30))

def timerange(start_time=8, end_time=20):
    """
    iterable for a range of times.
    default arguments are the standard hours for paid parking
    """
    for t in range(start_time,end_time):
        yield t