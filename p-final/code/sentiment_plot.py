#!/usr/bin/python

import argparse
import csv
from util_load import load
from util_load import clean_data
# from examples import clean_example

import numpy as np
import matplotlib
matplotlib.use('tkagg')    # For some reason this line is required to make pyplot work
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde



# Load data
reviews, header = load("wine_cleaned_google-final.csv", [14, 15, 4, 5])

sentiment = reviews[0]
magnitude = reviews[1]
points = reviews[2]
price = reviews[3]

# Remove wine reviews without a price
for i in range(len(sentiment) - 1, -1, -1):
    if price[i] == '':
        del sentiment[i]
        del magnitude[i]
        del points[i]
        del price[i]
        

'''
OPTIONS: Change what you want to plot here!!! ###
'''
X = sentiment
Y = points
xlabel = "Review Sentiment"
ylabel = "Score (Points)"
title = "Score (Points) v. Review Sentiment"

# Plot dimensions
alt_minmax = True # Set True to use new xmin/xmax and ymin/ymax as plot dimensions
xmin = -1
xmax = 1
ymin = 80
ymax = 100
'''
END OPTIONS
'''

# Plotting code
x = np.array(X).astype(np.float)
y = np.array(Y).astype(np.float)

k = gaussian_kde(np.vstack([x, y]))
xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

fig = plt.figure(figsize=(7,8))
# ax1 is the plot with boxes
ax1 = fig.add_subplot(211)
# ax2 is the topography map
ax2 = fig.add_subplot(212)


# alpha=0.5 will make the plots semitransparent
ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)


if alt_minmax:
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
else:
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(y.min(), y.max())
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(y.min(), y.max())


ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel)
ax2.set_xlabel(xlabel)
ax2.set_ylabel(ylabel)


plt.show()
