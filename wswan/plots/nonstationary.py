#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .common import GetBestRowsCols, calc_quiver, custom_cmap, bathy_cmap

# import constants
from .config import _faspect, _fsize, _fdpi

from ..storms import get_category


# supress matplotlib warnings
import warnings
warnings.filterwarnings("ignore")

# TODO:
#    PLOT VIDEO PANEL WINDS INPUT - VAR OUTPUT
# 
#    MEJORAR/COMPLETAR PLOTEOS
#    REFACTOR reordenar bien la libreria y la separacion: data / figura / axis-objeto 
#    REPASO IMPORTs
#    DOCUMENTAR 
#    STANDARD TITLE? 
#    STANDARD grafiti axsplot? 


# aux.functions

def mesh2np(mesh):
    'generates np depth from swan mesh'

    # TODO MESH.GET_DG_XY()
    depth = mesh.depth
    xpc = mesh.dg['xpc']
    ypc = mesh.dg['ypc']
    xlenc = mesh.dg['xlenc']
    ylenc = mesh.dg['ylenc']
    mxc = depth.shape[1]
    myc = depth.shape[0]
    XX = np.linspace(xpc, xpc+xlenc, mxc)
    YY = np.linspace(ypc, ypc+ylenc, myc)

    return XX, YY, depth

def get_storm_color(categ):

    dcs = {
        0 : 'lime',
        1 : 'yellow',
        2 : 'orange',
        3 : 'red',
        4 : 'purple',
        5 : 'black',
        6 : 'gray',
    }

    return dcs[categ]

def add_wind_module_dir(xds):
    'Calculate Wind_v(m/s) and Wind_dir(º) from windv_x,y'
    # TODO: llevar a extraer output points

    wx = xds['Windv_x']
    wy = xds['Windv_y']

    # module and dir
    ww = np.sqrt(np.power(wx, 2) + np.power(wy, 2))
    wd = np.degrees(np.arctan2(wy, wx))
    wd[wd<0] = wd[wd<0]+360  # TODO: direccion viento correcta?

    xds['Wind_v'] = xds['Windv_x'].copy()
    xds['Wind_v'][:] = ww
    xds['Wind_v'].attrs={'units':'m/s', 'description': 'Wind Speed'}

    xds['Wind_dir'] = xds['Windv_x'].copy()
    xds['Wind_dir'][:] = wd
    xds['Wind_dir'].attrs={'units':'º', 'description': 'Wind direction'}

    return(xds)

def storm_title(storm_track):
    'returns title for storm_track cases'

    t1 = 'Pmin: {0:.2f} hPa'.format(np.min(storm_track.p0))
    t2 = 'Vmean: {0:.2f} km/h'.format(np.mean(storm_track.vf)*1.852)

    ttl_st = '\n{0} / {1}'.format(t1, t2)

    if 'move' in storm_track:
        ttl_st += ' / Gamma: {0:.2f}º'.format(storm_track.move[0])

    return ttl_st


# axes generation

def axplot_shore(ax, np_shore):
    'adds a shore (numpy) to axes'

    ax.plot(
        np_shore[:,0], np_shore[:,1], '.', color='dimgray',
        markersize=3, label=''
    )

def axplot_labels(ax, coords_mode):
    'sets xlabel and ylabel from swan coords_mode'

    xlab, ylab = 'X (m)', 'Y (m)'
    if coords_mode == 'SPHERICAL':
        xlab, ylab = 'Longitude (º)', 'Latitude (º)'
    ax.set_xlabel(xlab, fontweight='bold')
    ax.set_ylabel(ylab, fontweight='bold')

def axplot_quiver(ax, XX, YY, vv, vd):
    'Plot 2D quiver map'

    size = 40
    scale = size / (XX[-1]-XX[0]) * np.nanmax(vv)
    if scale <0:    scale *= -1

    x_q, y_q, var_q, u, v = calc_quiver(XX, YY, vv, vd, size=size)

    ax.quiver(
        x_q, y_q, -u*var_q, -v*var_q,
        width=0.0015,
        scale = scale, #0.5,
        scale_units='x',
    )

def axplot_nested_meshes(ax, mesh_nested_list):
    'plot all ensted meshes'

    for mn in mesh_nested_list:
        XX_m, YY_m, _ = mesh2np(mn)
        xr = [XX_m[0], XX_m[-1], XX_m[-1], XX_m[0], XX_m[0]]
        yr = [YY_m[0], YY_m[0], YY_m[-1], YY_m[-1], YY_m[0]]

        ax.plot(xr, yr, '-', color='w', linewidth='4', label=mn.ID)

def axplot_var_map(ax, XX, YY, vv,
                   vmin = None, vmax = None,
                   cmap = plt.get_cmap('seismic'),
                  ):
    'plot 2D map with variable data'

    # cplot v lims
    if vmin == None: vmin = np.nanmin(vv)
    if vmax == None: vmax = np.nanmax(vv)

    # plot variable 2D map
    pm = ax.pcolormesh(
        XX, YY, vv,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        shading='auto',
        #zorder=0,
    )

    # fix axes
    ax.set_xlim(XX[0], XX[-1])
    ax.set_ylim(YY[0], YY[-1])

    # return pcolormesh
    return pm

def axplot_storm_track(ax, st, cat_colors=True):
    'plots storm track, category and target point'

    # storm trac parameters
    xt = st.lon
    yt = st.lat

    # plot track
    plt.plot(
        xt, yt, '-', linewidth=4,
        color='black', label='Storm Track'
    )

    # plot categories
    if cat_colors:

        # get category
        categ = np.array(get_category(st.p0))

        for c in range(7):
            lonc = xt[np.where(categ==c)[0]]
            latc = yt[np.where(categ==c)[0]]
            label = 'cat {0}'.format(c)
            if c==6: label = 'unknown'
            ax.plot(
                lonc, latc, '.', color=get_storm_color(c),
                markersize=10, label=label,
            )

    # target point
    if 'x0' in st.keys():
        ax.plot(
            st.x0, st.y0, '+', mew=3, ms=15,
            color='dodgerblue', label='target',
        )

def axplot_series(ax, xda_v, lc, mesh_ID, linestyle='-'):
    'axes plot variables series'

    # values and time
    vvs = xda_v.values[:]
    vts = xda_v.time.values[:]

    # plot series
    ax.plot(
        vts, vvs,
        linestyle=linestyle, linewidth=2, color=lc,
        label = mesh_ID,
    )

    ax.set_xlim(vts[0], vts[-1])


# INPUT Plots: 
#    - Project Site 
#    - Case Input 
#    - Vortex Input winds
#    - Vortex Grafiti winds

def plot_project_site(swan_proj, vmin=None, vmax=None, zoom=False, shoreline=False):
    '''
    Plots SwanProject site
        - bathymetry
        - control points
        - shoreline
        - nested meshes locations
    '''

    # figure
    fig, (axs) = plt.subplots(
        nrows=1, ncols=1,
        figsize=(_fsize*_faspect, _fsize),
    )

    # plot bathymetry
    mesh = swan_proj.mesh_main
    XX, YY, depth = mesh2np(mesh)

    if vmin == None: vmin = np.min(depth)
    if vmax == None: vmax = np.max(depth)

    pm = axplot_var_map(
        axs, XX, YY, -depth, vmin=vmin, vmax=vmax,#depth,
        cmap = bathy_cmap(np.abs(vmin), np.abs(vmax)),#'gist_earth_r',
    )
    cbar = fig.colorbar(pm, ax=axs)
    cbar.ax.set_ylabel('depth (m)', rotation=90, va="bottom", fontweight='bold')

    # mesh coordinates labels
    axplot_labels(axs, swan_proj.params['coords_mode'])

    # plot shoreline
    shore = swan_proj.shore
    if shore.any() and shoreline:
        axplot_shore(axs, np_shore=shore)

    # plot output points (control points)
    x_out = swan_proj.params['output_points_x']
    y_out = swan_proj.params['output_points_y']
    if x_out and len(x_out)==len(y_out):
        axs.plot(
            x_out, y_out, '.', color='red',
            markersize=16, label='control points'
        )

    # plot nested meshes
    axplot_nested_meshes(axs, swan_proj.mesh_nested_list)

    # plot zoom limits (nest0)
    if zoom:
        mesh0 = swan_proj.mesh_nested_list[0]
        XX_m, YY_m, _ = mesh2np(mesh0)
        axs.set_xlim([XX_m[0], XX_m[-1]])
        axs.set_ylim([YY_m[0], YY_m[-1]])

    # title
    axs.set_title('SWAN Project Site: {0}'.format(swan_proj.name),
                  fontsize=16, fontweight='bold')

    # turn on legend
    plt.legend(loc='upper right', prop={'size':10})

    # equal axis
    axs.set_aspect('equal', 'box')

    return fig

def plot_case_input(swan_proj, storm_track_list=[], case_number=0):
    '''
    Plots case input
        - boundary waves time series
        - TCs storm track
        - shoreline
        - control point
    '''

    # mesh values
    mesh = swan_proj.mesh_main
    XX, YY, _ = mesh2np(mesh)

    # figure
    fig, (axs) = plt.subplots(
        nrows=1, ncols=1,
        figsize=(_fsize*_faspect, _fsize),
    )

    # mesh coordinates labels
    axplot_labels(axs, swan_proj.params['coords_mode'])

    # plot shoreline
    shore = swan_proj.shore
    if shore.any():
        axplot_shore(axs, np_shore=shore)

    # TODO: add plot wave boundaries series

    # plot storm track
    if storm_track_list:
        st = storm_track_list[case_number]  # select storm track for this case
        axplot_storm_track(axs, st)

        # add text to title
        ttl_st = storm_title(st)

    else:
        ttl_st = ''

    # title
    axs.set_title(
        'SWAN Project: {0}, Case: {1:04d} {2}'.format(
            swan_proj.name, case_number, ttl_st),
        fontsize=16, fontweight='bold')

    # fix axes
    axs.set_xlim(XX[0], XX[-1])
    axs.set_ylim(YY[0], YY[-1])

    # legend
    plt.legend(loc='upper right', prop={'size':10})

    axs.set_facecolor('lightcyan')

    # equal axis
    axs.set_aspect('equal', 'box')

    return fig

def plot_case_vortex_input(swan_wrap, storm_track_list=[], t_num=10, case_number=0,
                            mesh=None, quiver=True, show_nested=False):
    '''
    Plots Case Vortex input
        - TCs storm track
        - TCs vortex winds
        - control points
        - shoreline
        - nested meshes locations (optional)
    '''

    # swan project
    swan_proj = swan_wrap.proj

    # default to main mesh
    if mesh == None: mesh = swan_proj.mesh_main

    # read vortex Wind module and direction
    case_id = '{0:04d}'.format(case_number)
    p_case = op.join(swan_proj.p_cases, case_id)
    code = 'wind_{0}'.format(mesh.ID)
    p_vortex = op.join(p_case, 'vortex_{0}.nc'.format(code))

    xds_vortex = xr.open_dataset(p_vortex)

    # select time to plot
    xds_v = xds_vortex.isel(time=t_num)

    # get mesh data from vortex dataset
    coords_mode = swan_proj.params['coords_mode']
    if coords_mode == 'SPHERICAL':      xa, ya = 'lon', 'lat'
    else:                               xa, ya = 'X', 'Y'

    X = xds_v[xa].values[:]
    Y = xds_v[ya].values[:]

    # vortex wind and dir
    xds_v_wnd = xds_v['W']
    xds_v_dir = xds_v['Dir']

    # figure
    fig, (axs) = plt.subplots(
        nrows=1, ncols=1,
        figsize=(_fsize*_faspect, _fsize),
    )

    # maximum and minimum wind values 
    vmax = float(xds_v_wnd.max().values)
    vmin = float(xds_v_wnd.min().values)
    wind_units = xds_v_wnd.units

    # plot vortex
    ccmap = custom_cmap(100, 'plasma_r', 0.05, 0.9, 'viridis', 0.2, 1)
    pm = axplot_var_map(
        axs, X, Y, xds_v_wnd,
        vmin = vmin, vmax = vmax,
        cmap = ccmap,
    )
    cbar = fig.colorbar(pm, ax=axs)
    cbar.ax.set_ylabel(
        '{0} ({1})'.format('Wind', wind_units),
        rotation=90, va="bottom", fontweight='bold',
        labelpad=15,
    )

    # plot quiver
    if quiver:
        axplot_quiver(axs, X, Y, xds_v_wnd.values[:], xds_v_dir.values[:])

    # plot shoreline
    shore = swan_proj.shore
    if shore.any():
        axplot_shore(axs, np_shore=shore)

    # plot storm track
    if storm_track_list:
        st = storm_track_list[case_number]  # select storm track for this case
        axplot_storm_track(axs, st, cat_colors=False)

    # mesh coordinates labels
    axplot_labels(axs, swan_proj.params['coords_mode'])

    # plot nested meshes
    if show_nested:
        axplot_nested_meshes(axs, swan_proj.mesh_nested_list)

    # title
    date_0 = xds_v.time.values
    fmt = '%d-%b-%Y %H:%M%p'
    t_str = pd.to_datetime(str(date_0)).strftime(fmt)
    ttl_t = '\nVortex Model, time: {0}'.format(t_str)
    axs.set_title(
        'SWAN Project: {0}, Case: {1:04d}, Mesh: {2} {3}'.format(
            swan_proj.name, case_number, mesh.ID, ttl_t),
        fontsize=16, fontweight='bold')

    # equal axis
    axs.set_aspect('equal', 'box')

    return fig

def plot_case_vortex_grafiti(swan_wrap, storm_track_list=[], case_number=0,
                             mesh=None, show_nested=False):
    '''
    Plots Case Vortex Grafiti
        - TCs storm track
        - TCs vortex Max. winds (Grafiti)
        - control points
        - shoreline
        - nested meshes locations (optional)
    '''

    # swan project
    swan_proj = swan_wrap.proj

    # default to main mesh
    if mesh == None: mesh = swan_proj.mesh_main

    # read vortex Wind module and direction
    case_id = '{0:04d}'.format(case_number)
    p_case = op.join(swan_proj.p_cases, case_id)
    code = 'wind_{0}'.format(mesh.ID)
    p_vortex = op.join(p_case, 'vortex_{0}.nc'.format(code))
    xds_vortex = xr.open_dataset(p_vortex)
    var_name = 'W'


    # figure
    fig, (axs) = plt.subplots(
        nrows=1, ncols=1,
        figsize=(_fsize*_faspect, _fsize),
    )

    # plot grafiti
    xds_var = xds_vortex[var_name]
    var_units = xds_var.units

    # get mesh data from output dataset
    coords_mode = swan_proj.params['coords_mode']
    if coords_mode == 'SPHERICAL':
        xa, ya = 'lon', 'lat'
    else:
        xa, ya = 'X', 'Y'
    X = xds_var[xa].values[:]
    Y = xds_var[ya].values[:]

    # maximum and minimum values 
    vmax = float(xds_var.max().values)
    vmin = float(xds_var.min().values)

    # grafiti
    xds_var_max = xds_var.max(dim='time')
    var_max = xds_var_max.values[:]

    ccmap = custom_cmap(100, 'plasma_r', 0.05, 0.9, 'viridis', 0.2, 1)
    pm = axplot_var_map(
        axs, X, Y, var_max,
        vmin = vmin, vmax = vmax,
        cmap = ccmap,
    )
    cbar = fig.colorbar(pm, ax=axs)
    cbar.ax.set_ylabel(
        '{0} max. ({1})'.format(var_name, var_units),
        rotation=90, va="bottom", fontweight='bold',
        labelpad=15,
    )

    # plot shoreline
    shore = swan_proj.shore
    if shore.any():
        axplot_shore(axs, np_shore=shore)

    # plot storm track
    if storm_track_list:
        st = storm_track_list[case_number]  # select storm track for this case
        axplot_storm_track(axs, st, cat_colors=False)

    # mesh coordinates labels
    axplot_labels(axs, swan_proj.params['coords_mode'])

    # plot nested meshes
    if show_nested:
        axplot_nested_meshes(axs, swan_proj.mesh_nested_list)

    # title
    ttl_t = '\nVortex Model, Grafiti Max. Winds'
    axs.set_title(
        'SWAN Project: {0}, Case: {1:04d}, Mesh: {2} {3}'.format(
            swan_proj.name, case_number, mesh.ID, ttl_t),
        fontsize=16, fontweight='bold')

    # equal axis
    axs.set_aspect('equal', 'box')

    return fig

# TODO revisar
def plot_matrix_input(swan_proj, storm_track_list=[], 
                      case_ini=0, case_end=9, mesh=None, show=True):
    '''
    # TODO: documentar
    '''
    # TODO: refactor con el siguiente grafiti

    # default to main mesh
    if mesh == None: mesh = swan_proj.mesh_main
    XX, YY, _ = mesh2np(mesh)

    # get number of rows and cols for gridplot 
    n_clusters = np.arange(case_ini, case_end).size
    n_rows, n_cols = GetBestRowsCols(n_clusters)

    # figure
    fig = plt.figure(figsize=(23*1.5, 20*1.5))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0, hspace=0)
    gr, gc = 0, 0

    for case_i in range(case_ini, case_end):

        # plot variable times
        ax = plt.subplot(gs[gr, gc])

        # plot shoreline
        shore = swan_proj.shore
        if shore.any():
            axplot_shore(ax, np_shore=shore)

        # plot storm track
        if storm_track_list:
            st = storm_track_list[case_i]  # select storm track for this case
            axplot_storm_track(ax, st)

            # add text to title
            ttl_st = storm_title(st)
            ax.text(
                0.1, 0.02, '{0}'.format(ttl_st),
                color='k', fontsize=18,
                transform=ax.transAxes
            )

        # number
        ax.text(0.02, 0.02, case_i, color='fuchsia', fontweight='bold', fontsize=20,
                transform=ax.transAxes)

        # fix axes
        ax.set_xlim(XX[0], XX[-1])
        ax.set_ylim(YY[0], YY[-1])

        # legend
        plt.legend(loc='upper right', prop={'size':10})

        ax.set_facecolor('lightcyan')

        # equal axis
        ax.set_aspect('equal', 'box')
        #ax.axis('off')

        # counter
        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    # show and return figure
    if show: plt.show()


# OUTPUT Plots (.mat file, 2D data): 
#    - Case output
#    - Case output grafiti
#    - Case output points (TABLE)

def plot_case_output(
    swan_wrap, var_name='Hsig', case=0, mesh=None,
    storm_track_list = [], t_num=0, quiver=True, show_nested=False):
    '''
    Plots Case 2D output (.mat file)
        - 2D output variable "var_name"
        - TCs storm track
        - control points
        - shoreline
        - nested meshes locations (optional)
    '''

    # swan project
    swan_proj = swan_wrap.proj

    # default to main mesh
    if mesh == None: mesh = swan_proj.mesh_main

    # load output
    xds_out = swan_wrap.extract_output(
        case_ini=case, case_end=case+1,
        mesh=mesh, concat=False,
    )[0]

    # select time to plot
    xds_v = xds_out.isel(time=t_num)

    # variable to plot
    xds_var = xds_v[var_name]
    var_units = xds_var.units

    # get mesh data from output dataset
    coords_mode = swan_proj.params['coords_mode']
    if coords_mode == 'SPHERICAL':
        xa, ya = 'lon', 'lat'
    else:
        xa, ya = 'X', 'Y'
    X = xds_var[xa].values[:]
    Y = xds_var[ya].values[:]

    # figure
    fig, (axs) = plt.subplots(
        nrows=1, ncols=1,
        figsize=(_fsize*_faspect, _fsize),
    )

    # maximum and minimum values 
    vmax = float(xds_var.max().values)
    vmin = float(xds_var.min().values)

    # plot output variable
    ccmap = custom_cmap(15, 'YlOrRd', 0.15, 0.9, 'YlGnBu_r', 0, 0.85)
    pm = axplot_var_map(
        axs, X, Y, xds_var.values[:],
        vmin = vmin, vmax = vmax,
        cmap = ccmap,
    )
    cbar = fig.colorbar(pm, ax=axs)
    cbar.ax.set_ylabel(
        '{0} ({1})'.format(var_name, var_units),
        rotation=90, va="bottom", fontweight='bold',
        labelpad=15,
    )

    # plot quiver
    if quiver:
        axplot_quiver(axs, X, Y, xds_var.values[:], xds_v.Dir.values[:])

    # plot shoreline
    shore = swan_proj.shore
    if shore.any():
        axplot_shore(axs, np_shore=shore)

    # plot storm track
    if storm_track_list:
        st = storm_track_list[case]  # select storm track for this case
        axplot_storm_track(axs, st, cat_colors=False)

    # mesh coordinates labels
    axplot_labels(axs, swan_proj.params['coords_mode'])

    # plot nested meshes
    if show_nested:
        axplot_nested_meshes(axs, swan_proj.mesh_nested_list)

    # title
    date_0 = xds_var.time.values
    fmt = '%d-%b-%Y %H:%M%p'
    t_str = pd.to_datetime(str(date_0)).strftime(fmt)
    ttl_t = '\n({0})'.format(t_str)
    axs.set_title(
        'SWAN Project: {0}, Case: {1:04d}, Mesh: {2} {3}'.format(
            swan_proj.name, case, mesh.ID, ttl_t),
        fontsize=16, fontweight='bold')

    # equal axis
    axs.set_aspect('equal', 'box')

    return fig

def plot_case_output_grafiti(
    swan_wrap, var_name='Hsig', case=0, mesh=None,
    storm_track_list = [], show_nested=False):
    '''
    Plots Case 2D output (.mat file) Grafiti Max.
        - 2D output variable "var_name" Max
        - TCs storm track
        - control points
        - shoreline
        - nested meshes locations (optional)
    '''

    # swan project
    swan_proj = swan_wrap.proj

    # default to main mesh
    if mesh == None: mesh = swan_proj.mesh_main

    # load output
    xds_out = swan_wrap.extract_output(
        case_ini=case, case_end=case+1,
        mesh=mesh, concat=False,
    )[0]

    # figure
    fig, (axs) = plt.subplots(
        nrows=1, ncols=1,
        figsize=(_fsize*_faspect, _fsize),
    )

    # plot grafiti
    xds_var = xds_out[var_name]
    var_units = xds_var.units

    # get mesh data from output dataset
    coords_mode = swan_proj.params['coords_mode']
    if coords_mode == 'SPHERICAL':      xa, ya = 'lon', 'lat'
    else:                               xa, ya = 'X', 'Y'
    X = xds_var[xa].values[:]
    Y = xds_var[ya].values[:]

    # maximum and minimum values 
    vmax = float(xds_var.max().values)
    vmin = float(xds_var.min().values)

    # grafiti
    xds_var_max = xds_var.max(dim='time')
    var_max = xds_var_max.values[:]

    ccmap = custom_cmap(15, 'YlOrRd', 0.15, 0.9, 'YlGnBu_r', 0, 0.85)
    pm = axplot_var_map(
        axs, X, Y, var_max,
        vmin = vmin, vmax = vmax,
        cmap = ccmap,
    )
    cbar = fig.colorbar(pm, ax=axs)
    cbar.ax.set_ylabel(
        '{0} max. ({1})'.format(var_name, var_units),
        rotation=90, va="bottom", fontweight='bold',
        labelpad=15,
    )

    # plot shoreline
    shore = swan_proj.shore
    if shore.any():
        axplot_shore(axs, np_shore=shore)

    # plot storm track
    if storm_track_list:
        st = storm_track_list[case]  # select storm track for this case
        axplot_storm_track(axs, st, cat_colors=False)

    # mesh coordinates labels
    axplot_labels(axs, swan_proj.params['coords_mode'])

    # plot nested meshes
    if show_nested:
        axplot_nested_meshes(axs, swan_proj.mesh_nested_list)

    # title
    date_0 = xds_out.time.values[0]
    date_1 = xds_out.time.values[-1]
    fmt = '%d-%b-%Y %H:%M%p'
    t_str_ini = pd.to_datetime(str(date_0)).strftime(fmt)
    t_str_fin = pd.to_datetime(str(date_1)).strftime(fmt)
    ttl_t = '\nGrafiti Max ({0} : {1})'.format(t_str_ini, t_str_fin)
    axs.set_title(
        'SWAN Project: {0}, Case: {1:04d}, Mesh: {2} {3}'.format(
            swan_proj.name, case, mesh.ID, ttl_t),
        fontsize=16, fontweight='bold')

    # equal axis
    axs.set_aspect('equal', 'box')

    return fig

def plot_case_output_points(swan_wrap, point=0, case=0):
    '''
    Plots Case time series output (TABLE points).
        - Plot all available variables at TABLE output
        - Plot main and all nested meshes
        - Plot needs to sel point and case
    '''

    # extract output from main mesh
    xds_main = swan_wrap.extract_output_points(
        case_ini = case, case_end = case+1,
        mesh = swan_wrap.proj.mesh_main,
    ).sel(case=case, point=point)

    # extract output from nested meshes
    l_xds_nest = [
        swan_wrap.extract_output_points(
            case_ini = case, case_end = case+1, mesh = mn,
        ).sel(case=case, point=point) for mn in swan_wrap.proj.mesh_nested_list
    ]

    # TODO: mover a extraer output points
    if 'Windv_x' in xds_main.variables and 'Windv_y' in xds_main.variables:
        xds_main = add_wind_module_dir(xds_main)
        l_xds_nest = [add_wind_module_dir(x) for x in l_xds_nest]

    # vars to plot
    block = ['x_point', 'y_point', 'time', 'DEPTH', 'OUT', 'case']
    vns = [v for v in xds_main.variables if v not in block]
    n_axis = len(vns)

    # figure
    fig, (axs) = plt.subplots(
        nrows=n_axis, ncols=1,
        figsize=(_fsize*_faspect, (_fsize/6)*n_axis),
        sharex = True,
    )

    nm_cs = ['r'] * len(l_xds_nest)  # TODO each nested mesh output line plot color

    # one axes for each variable
    for c, vn in enumerate(vns):

            # get variable units
            vu = xds_main[vn].attrs['units']
            lns = 'dotted' if vu == 'º' else '-'

            # plot main mesh 
            axplot_series(
                axs[c], xds_main[vn], 'black',
                xds_main.attrs['mesh_ID'], linestyle=lns,
            )

            # plot nestes meshes
            for nm, nmc in zip(l_xds_nest, nm_cs):
                axplot_series(
                    axs[c], nm[vn], nmc, nm.attrs['mesh_ID'], linestyle=lns,
                )

            # customize axes and labels
            axs[c].set_ylabel('{0} ({1})'.format(vn, vu),
                              rotation=90, fontweight='bold', labelpad=35)

            # fix dir axis
            if vu=='º':
                axs[c].set_ylim([0, 360])

    # add legend
    axs[0].legend()

    return fig

def axplot_grafiti(ax, swan_proj, xds_case, case_number, var_name,
                   storm_track_list=[], vmin=None, vmax=None):
    '''
    # TODO: documentar
    '''

    # plot grafiti
    xds_var = xds_case[var_name]

    # get mesh data from output dataset
    coords_mode = swan_proj.params['coords_mode']
    if coords_mode == 'SPHERICAL':      xa, ya = 'lon', 'lat'
    else:                               xa, ya = 'X', 'Y'
    X = xds_var[xa].values[:]
    Y = xds_var[ya].values[:]

    # grafiti
    xds_var_max = xds_var.max(dim='time')
    var_max = xds_var_max.values[:]

    # colormap
    if var_name=='W':    ccmap = custom_cmap(100, 'plasma_r', 0.05, 0.9, 'viridis', 0.2, 1)
    if var_name=='Hsig': ccmap = custom_cmap(100, 'YlOrRd', 0.09, 0.9, 'YlGnBu_r', 0, 0.88)

    pc = axplot_var_map(
        ax, X, Y, var_max,  # TODO output.T, aqui no
        vmin = vmin, vmax = vmax,
        cmap = ccmap,
    )

    # plot shoreline
    shore = swan_proj.shore
    if shore.any():
        axplot_shore(ax, np_shore=shore)

    # plot storm track
    if storm_track_list:
        st = storm_track_list[case_number]  # select storm track for this case
        axplot_storm_track(ax, st, cat_colors=False)

    # number
    ax.text(0.02, 0.02, case_number, color='fuchsia', fontweight='bold', fontsize=20,
            transform=ax.transAxes)

    plt.axis('scaled')
    plt.xlim(X[0], X[-1])
    plt.ylim(Y[0], Y[-1])
    plt.axis('off')

    # equal axis
#    ax.set_aspect('equal', 'box')

    return pc

def plot_matrix_grafiti(swan_wrap, var_name, storm_track_list=[],
                        case_ini=None, case_end=None, mesh=None, 
                        width=23*1.5, height=20*1.5):
    '''
    # TODO: documentar
    var_name: 'W' (vortex), 'Hsig' (output)
    '''

    # swan project
    swan_proj = swan_wrap.proj

    # default to main mesh
    if mesh == None: mesh = swan_proj.mesh_main

    if var_name == 'W':
        # read vortex Wind module and direction
        code = 'wind_{0}'.format(mesh.ID)

        l_xds_out = []
        for i in np.arange(case_ini, case_end):
            case_id = '{0:04d}'.format(i)
            p_case = op.join(swan_proj.p_cases, case_id)
            p_vortex = op.join(p_case, 'vortex_{0}.nc'.format(code))

            xds_vortex = xr.open_dataset(p_vortex)
            xds_vortex['case'] = i

            l_xds_out.append(xds_vortex)

    if var_name == 'Hsig':
        # load output list
        l_xds_out = swan_wrap.extract_output(
            mesh = mesh,
            case_ini = case_ini, case_end = case_end,
            var_name = var_name,
        )

    # extract min,max colobar limits
    var_max_all, var_min_all, n_clusters = [], [], []
    for xds_out in l_xds_out:
        var_max_all.append(xds_out[var_name].max(axis=0).max())
        var_min_all.append(xds_out[var_name].min(axis=0).min())
    var_max = np.nanmax(var_max_all)
    var_min = np.nanmin(var_min_all)

    # get number of rows and cols for gridplot 
    n_rows, n_cols = GetBestRowsCols(len(l_xds_out))

    ###########################################################################
    if n_cols==1:    # when not an integer or divisor !!!
        sqrt_n = np.sqrt(len(l_xds_out)).round()
        n_rows = int(sqrt_n)
        n_cols = int(sqrt_n)
    if n_rows > n_cols: 
        n_rows_0 = n_rows
        n_rows = n_cols
        n_cols = n_rows_0

    ###########################################################################

    # figure
    fig = plt.figure(figsize=(width, height))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0, hspace=0)
    gr, gc = 0, 0

    for xds_out in l_xds_out:

        # select case
        var_units = xds_out[var_name].attrs['units']

        # plot variable times
        ax = plt.subplot(gs[gr, gc])
        pc = axplot_grafiti(
            ax, swan_proj, xds_out, int(xds_out.case), var_name,
            storm_track_list=storm_track_list,
            vmin=var_min, vmax=var_max
        )

        # get lower positions
        if gr==n_rows-1 and gc==0:
            pax_l = ax.get_position()
        elif gr==n_rows-1 and gc==n_cols-1:
            pax_r = ax.get_position()

        # counter
        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    ###########################################################################
#    if gc <= n_cols: 
#        while gc < n_cols:
#            print(gc)
#            #ax = plt.subplot(gs[gr, gc])
#           # ax.axis('off')
#            gc += 1
    ###########################################################################

    cbar_ax = fig.add_axes([pax_l.x0, pax_l.y0-0.05, pax_r.x1 - pax_l.x0, 0.02])
    cb = fig.colorbar(pc, cax=cbar_ax, orientation='horizontal')
    cb.set_label(label='{0} ({1})'.format(var_name, var_units), size=20, weight='bold')
    cb.ax.tick_params(labelsize=15)

    return fig

