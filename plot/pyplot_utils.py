"""Module for various pylot utils."""

import numpy as np
import matplotlib.pyplot as plt


def histogram_plotter(ax, x_labels, y_labels, width=0.8):
    """Plot histogram on the given axes."""
    n = len(x_labels)
    tickLocations = np.arange(n)
    rectLocations = tickLocations - (width / 2.0)
    rects = ax.bar(rectLocations,
           y_labels,
           width)
    label_rect(ax, rects)
    ax.set_xticks(ticks=tickLocations)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(min(tickLocations) - 0.6, max(tickLocations) + 0.6)
    ax.set_yticks(np.arange(y_labels.min(), y_labels.max(), int(y_labels.std())))
    ax.yaxis.grid(True)


def label_rect(ax, rects):
    """Attach text to bar chart rectangles."""
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom', rotation=90)
