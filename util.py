"""util.py

General utilities

"""

import os

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
