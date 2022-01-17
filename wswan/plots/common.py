#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import cmocean


def GetDivisors(x):
    l_div = []
    i = 1
    while i<x:
        if x%i == 0:
            l_div.append(i)
        i = i + 1
    return l_div

def GetBestRowsCols(n):
    'try to square number n, used at gridspec plots'

    sqrt_n = sqrt(n)
    if sqrt_n.is_integer():
        n_r = int(sqrt_n)
        n_c = int(sqrt_n)
    else:
        l_div = GetDivisors(n)
        n_c = l_div[int(len(l_div)/2)]
        n_r = int(n/n_c)

    return n_r, n_c

def calc_quiver(X, Y, var, vdir, size=30):
    '''
    interpolates var and plots quiver with var_dir. Requires open figure

    X, Y - mesh grid dim. arrays
    var  - variable module
    vdir - variable direction (º clockwise relative to North)

    opt. args
    size - quiver mesh size

    **Spherical direction convention: direction clockwise from geographic north

    returns data for quiver plot (x_q, y_q, var_q, u, v)
        then plot with: plt.quiver(x_q, y_q, -u*var_q, -v*var_q)
    '''

    step = (X[-1]-X[0]) / size

    size_x = round((X[-1]-X[0]) / step)
    size_y = round((Y[-1]-Y[0]) / step)

    # var and dir interpolators 
    vdir_f = vdir.copy()
    vdir_f[np.isnan(vdir_f)] = 0
    vdir_f_y = np.cos(np.deg2rad(vdir_f))
    vdir_f_x = np.sin(np.deg2rad(vdir_f))
    f_dir_x = interpolate.interp2d(X, Y, vdir_f_x, kind='linear')
    f_dir_y = interpolate.interp2d(X, Y, vdir_f_y, kind='linear')

    var_f = var.copy()
    var_f[np.isnan(var_f)] = 0
    f_var = interpolate.interp2d(X, Y, var_f, kind='linear')

    # generate quiver mesh
    x_q = np.linspace(X[0], X[-1], num = size_x)
    y_q = np.linspace(Y[0], Y[-1], num = size_y)

    # interpolate data to quiver mesh
    var_q = f_var(x_q, y_q)
    vdir_q_x = f_dir_x(x_q, y_q)
    vdir_q_y = f_dir_y(x_q, y_q)
    vdir_q = np.rad2deg(np.arctan(vdir_q_x/vdir_q_y))
    # sign correction
    vdir_q[(vdir_q_x>0) & (vdir_q_y<0)] = vdir_q[(vdir_q_x>0) & (vdir_q_y<0)] + 180
    vdir_q[(vdir_q_x<0) & (vdir_q_y<0)] = vdir_q[(vdir_q_x<0) & (vdir_q_y<0)] + 180

    # u and v dir components
    u = np.sin(np.deg2rad(vdir_q))
    v = np.cos(np.deg2rad(vdir_q))

    return x_q, y_q, var_q, u, v


# aux matplotlib utils

def custom_cmap(numcolors, map1, m1ini, m1end, map2, m2ini, m2end):
    '''
    Generate custom colormap
    Example: Red-Orange-Yellow-Green-Blue -- map1='YlOrRd' map2='YlGnBu_r'
    mXini, mXend:   colormap range of colors
    numcolors:      number of colors (100-continuous, 15-discretization)
    '''

    # color maps
    cmap1 = plt.get_cmap(map1, numcolors)
    cmap2 = plt.get_cmap(map2, numcolors)

    # custom color ranges
    cmap1v = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                n=cmap1.name, a=m1ini, b=m1end),
            cmap1(np.linspace(m1ini,m1end,100))
    )

    cmap2v = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                n=cmap2.name, a=m2ini, b=m2end),
            cmap2(np.linspace(m2ini,m2end,100))
    )

    top = cm.get_cmap(cmap1v, 128)
    bottom = cm.get_cmap(cmap2v, 128)

    newcolors = np.vstack((
        bottom(np.linspace(0,1,128)),
        top(np.linspace(0,1,128))
    ))
    newcmp = colors.ListedColormap(newcolors, name='OrangeBlue')

    return newcmp

def bathy_cmap(bottom_lim, top_lim):
    '''
    Generate custom colormap for bathymetry plots
    '''

    # colormaps
    colors1 = cmocean.cm.turbid
    colors2 = 'YlGnBu_r'

    top = cm.get_cmap(colors1, top_lim)
    bottom = cm.get_cmap(colors2, bottom_lim)

    newcolors = np.vstack((bottom(np.linspace(0, 0.8, bottom_lim)),
                       top(np.linspace(0.1, 1, top_lim))))

    return (colors.ListedColormap(newcolors))

