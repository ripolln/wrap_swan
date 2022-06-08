#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

from .common import GetBestRowsCols, calc_quiver

# import constants
from .config import _faspect, _fsize, _fdpi


def axplot_var_map(ax, XX, YY, vv, vd,
                   quiver=True, np_shore=np.array([]),
                   vmin=None, vmax=None, cmap=None,
                   remove_axis=False):
    'plot 2D map with variable data'

    # parameters
    if cmap == None:
        cmap = plt.get_cmap('seismic')

    # cplot v lims
    if vmin == None: vmin = vv.nanmin()
    if vmax == None: vmax = vv.nanmax()

    # plot variable 2D map
    pm = ax.pcolormesh(
        XX, YY, vv,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        shading='auto',
    )

    # optional quiver
    if quiver:
        x_q, y_q, var_q, u, v = calc_quiver(XX[0,:], YY[:,0], vv, vd, size=12)
        ax.quiver(
            x_q, y_q, -u*var_q, -v*var_q,
            width=0.003,
            #scale = 0.5,
            scale_units='inches',
        )

    # optional shoreline
    if np_shore.any():
        xs = np_shore[:,0]
        ys = np_shore[:,1]
        ax.plot(
            np_shore[:,0], np_shore[:,1],
            '.', color='dimgray',
            markersize=3, label=''
        )

        # fix axes
        ax.set_xlim(XX[0,0], XX[0,-1])
        ax.set_ylim(YY[0,0], YY[-1,0])

    if remove_axis:
        ax.set_xticks([])
        ax.set_yticks([])


    # return last pcolormesh
    return pm

def scatter_maps(xds_out, var_list=[], n_cases=None, n_cols=None, n_rows=None,
                 quiver=True, var_limits={}, np_shore=np.array([]),
                 figsize=None):
    '''
    scatter plots stationary SWAN execution output for first "n_cases"

    xds_out    - swan stationary output (xarray.Dataset)

    opt. args
    var_list   - swan output variables ['Hsig', 'Tm02', 'Tpsmoo'] (default all vars)
    n_cases    - number of cases to plot (default all cases)
    quiver     - True for adding directional quiver plot
    var_limits  - dictionary with variable names as keys and a (min, max) tuple for limits
    np_shore   - shoreline, np.array x = np_shore[:,0] y = np.shore[:,1]
    '''

    # TODO improve xticks, yticks 
    # TODO legend box with info ?

    # number of cases
    if n_cases == None:
        n_cases = len(xds_out.case.values)

    # get number of rows and cols for gridplot
    if n_rows == None or n_cols == None:
        n_cols, n_rows = GetBestRowsCols(n_cases)

    # figure size
    if figsize == None:
        figsize = (_fsize*_faspect, _fsize*_faspect)

    # allowed vars
    avs = ['Hsig', 'Tm02', 'Dspr', 'TPsmoo', 'Dir', 'Tp']

    # colormap dictionary
    cmap_dict = {
        'Dir': 'twilight_shifted',
        'Hsig': 'RdBu_r',
        'Tm02': 'magma_r',
        'Tpsmoo': 'magma_r',
        'Tp': 'magma_r',
        'Dspr': 'rainbow',
    }

    # TODO check samoa project then delete
    #    if wind:
    #        if vn =='Hsig': 
    #            cmap='inferno_r'
    #    else:
    #        if vn =='Hsig': 
    #            cmap='RdBu_r'
    #    if vn== 'count_parts':
    #        cmap='rainbow'


    # variable list 
    if var_list == []:
        var_list = dict(xds_out.variables).keys()
        var_list = [vn for vn in var_list if vn in avs]  # filter only allowed

    # mesh data to plot
    if 'lon' in xds_out.dims:
        X, Y = xds_out.lon, xds_out.lat
        xlab, ylab = 'Longitude (º)', 'Latitude (º)'
    else:
        X, Y = xds_out.X, xds_out.Y
        xlab, ylab = 'X', 'Y'
    XX, YY = np.meshgrid(X, Y)

    # iterate: one figure for each variable
    l_figs = []
    for vn in var_list:

        # plot figure
        fig, (axs) = plt.subplots(
            nrows=n_rows, ncols=n_cols,
            sharex=True, sharey=True,
            constrained_layout=False,
            figsize=figsize,
        )

        fig.subplots_adjust(wspace=0, hspace=0)

        # common vlimits
        vmin = xds_out[vn].min()
        vmax = xds_out[vn].max()

        # optional vlimits
        if vn in var_limits.keys():
            vmin = var_limits[vn][0]
            vmax = var_limits[vn][1]

        # plot cases output
        gr, gc = 0, 0
        for ix in range(n_cases):
            out_case = xds_out.isel(case=ix)

            # variable and direction 
            vv = out_case[vn].values[:]
            vd = out_case['Dir'].values[:]

            # plot variable times
            ax = axs[gr, gc]
            pm = axplot_var_map(
                ax, XX, YY, vv, vd,
                quiver=quiver, np_shore=np_shore,
                vmin=vmin, vmax=vmax,
                cmap = cmap_dict[vn],
                remove_axis = True,
            )

            # row,col counter
            gc += 1
            if gc >= n_cols:
                gc = 0
                gr += 1


        # add custom common axis labels
        fig.text(0.5, 0.04, xlab, ha='center', fontsize=15)
        fig.text(0.04, 0.5, ylab, va='center', rotation='vertical', fontsize=15)

        # add custom common colorbar
        cbar_ax = fig.add_axes([0.93, 0.11, 0.02, 0.77])
        fig.colorbar(pm, cax=cbar_ax)
        cbar_ax.set_ylabel(vn, fontsize=15)


        l_figs.append(fig)

    return l_figs

