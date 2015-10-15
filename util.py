"""util.py

General utilities

"""

import os
import sys

def get_or_default(d, k, default_v):
    try:
        return d[k]
    except KeyError:
        return default_v

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) 

def last_equal(lst):
    """Return longest sequence of equal elements at the end of a list.

    >>> last_equal([])
    []
    >>> last_equal([1, 2, 3, 3, 3])
    [3, 3, 3]
    >>> last_equal([1, 2, 3])
    [3]

    """
    els = []
    for v in reversed(lst):
        if len(els) == 0 or v == els[0]:
            els.append(v)
        else:
            break
    return els

def last_true(lst, f):
    """Return longest sequence of elements at end of list that pass f.

    >>> last_true([], lambda x: x % 2 == 1)
    []
    >>> last_true([1, 2, 3, 3, 3], lambda x: x % 2 == 1)
    [3, 3, 3]
    >>> last_true([1, 2, 3], lambda x: x % 2 == 1)
    [3]
    >>> last_true([1, 2, 3], lambda x: x % 2 == 0)
    []

    """
    els = last_equal(map(f, lst))
    if len(els) == 0 or els[0]:
        return lst[-1 * len(els):]
    else:
        return []


#--------- Statistics ------

from scipy.special import gamma
import scipy.stats as ss

beta = ss.beta.pdf

def dbeta(x, a, b):
    return gamma(a+b)/(gamma(a)*gamma(b)) * \
           ((a-1) * x**(a-2) * (1-x)**(b-1) - x**(a-1) * (b-1) * (1-x)**(b-2))

assert dbeta(0.5, 2, 2) == 0
assert dbeta(0.6, 2, 2) != 0

def dirichlet_mode(x):
    return [(v - 1) / (sum(x) - len(x)) for v in x]

def beta_fit(mode, mag):
    """Return parameters for beta distribution with given mode.

    Solves a system of linear equations to find alpha and beta s.t.
    alpha + beta = mag;
    (alpha - 1) / (alpha + beta - 2) = mode.

    Args:
        mode:  Mode of desired beta distribution.
        mag:   Magnitude of alpha + beta.

    Returns:
        x: Numpy array of [alpha, beta]

    >>> beta_fit(0.1, 7)
    array([ 1.5,  5.5])
    >>> beta_fit(0.2, 7)
    array([ 2.,  5.])
    >>> beta_fit(0.3, 7)
    array([ 2.5,  4.5])

    """
    a = np.array([[mode - 1, mode],
                  [1, 1]])
    b = np.array([2 * mode  - 1, mag])
    x = np.linalg.solve(a, b)
    return x



#---------- Plotting ---------

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

def savefig(ax, name):
    """Save the current figure taking into account legends."""
    lgd = ax.legend()
    if lgd is not None:
        plt.savefig(name, bbox_inches='tight', bbox_extra_artists=(lgd,))
    else:
        plt.savefig(name, bbox_inches='tight')

def tsplot_robust(df, time, unit, condition, value, ci=95,
                  logx=False, logy=False):
    """Plot timeseries data with different x measurements.

    Returns:
        ax:     Axis object.
        df:     Dataframe containing: time, condition, mean, n, sem.

    """
    n = df.groupby([condition, time])[value].count()
    n.name = 'n'
    means = df.groupby([condition, time])[value].mean()
    means.name = 'mean'
    sem = df.groupby([condition, time])[value].aggregate(ss.sem)
    sem.name = 'sem'
    df_stat = pd.concat([means, n, sem], axis=1).reset_index()

    # Use seaborn iff all conditions have the same number of measurements for
    # each time point.
    n_pivot = n.reset_index().pivot(index=time, columns=condition, values='n')
    if len(df) == 0:
        raise Exception('Unable to plot empty dataframe')
    if len(n_pivot) == 1:
        ax = sns.barplot(condition, y=value, data=df, ci=ci)
    elif len(n_pivot) == sum(n_pivot.duplicated()) + 1:
        ax = sns.tsplot(df, time=time, condition=condition,
                        unit=unit, value=value, ci=ci)
    else:
        if ci != 95:
            raise NotImplementedError
        ax = plt.gca()
        for c, df in df_stat.groupby(condition):
            line, = plt.plot(df[time], df['mean'], label=c)
            # Disable filling for logy, since may exceed desirable range.
            if not logy:
                ax.fill_between(df[time],
                                df['mean'] - 1.96 * df['sem'],
                                df['mean'] + 1.96 * df['sem'],
                                facecolor=line.get_color(),
                                alpha=0.5,
                                where=np.isfinite(df['sem']))
        plt.legend()
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    return ax, df_stat

#-----------  Multiprocessing -------

import traceback
import multiprocessing as mp
import re

def init_worker():
    """Function to make sure everyone happily exits on KeyboardInterrupt

    See https://stackoverflow.com/questions/1408356/
    keyboard-interrupts-with-pythons-multiprocessing-pool

    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def run_functor(functor, x):
    """
    Given a no-argument functor, run it and return its result. We can
    use this with multiprocessing.map and map it over a list of job
    functors to do them.

    Handles getting more than multiprocessing's pitiful exception output

    https://stackoverflow.com/questions/6126007/
    python-getting-a-traceback-from-a-multiprocessing-process

    """
    try:
        # This is where you do your actual work
        return functor(x)
    except:
        # Put all exception text into an exception and raise that
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))

def cpu_count():
    """Return number of cpus respecting numactl restrictions.

    Never exceeds number of cpus specified by mp.cpu_count().

    """
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return min(res, mp.cpu_count())
    except IOError:
        return mp.cpu_count()
