#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import sys
import os
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr

# dev
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# swan wrap module
from wswan.wrap import SwanProject, SwanMesh, SwanWrap_NONSTAT, SwanInput_NONSTAT

# TODO: programar vientos custom 2D con mallas anidadas

# --------------------------------------
# data
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))


# --------------------------------------
# SWAN project (config bathymetry, parameters, computational grid)

p_proj = op.join(p_data, 'projects')  # swan projects main directory
n_proj = '07_wind_2d'      # project name

sp = SwanProject(p_proj, n_proj)


# --------------------------------------
# SWAN main mesh
dx = 4000000
dy = 500000
xlon = np.arange(0,dx+1, 20000)
ylat = np.arange(0,dy+1, 20000)

# SWAN main mesh
main_mesh = SwanMesh()

# depth grid description (input bathymetry grid)
main_mesh.dg = {
    'xpc': 0,       # x origin
    'ypc': 0,       # y origin
    'alpc': 0,      # x-axis direction 
    'xlenc': xlon[-1] - xlon[0],   # grid length in x
    'ylenc': ylat[-1] - ylat[0],  # grid length in y
    'mxc': 1,       # number mesh x
    'myc': 1,       # number mesh y
    'dxinp': xlon[-1] - xlon[0],   # size mesh x
    'dyinp': ylat[-1] - ylat[0],  # size mesh y
}

# depth value
main_mesh.depth = np.ones((2,2)) * 200

# computational grid description
main_mesh.cg = {
    'xpc': 0,
    'ypc': 0,
    'alpc': 0,
    'xlenc': xlon[-1] - xlon[0],
    'ylenc': ylat[-1] - ylat[0],
    'mxc': xlon.size-1,
    'myc': ylat.size-1,
    'dxinp': (xlon[-1] - xlon[0])/(xlon.size-1),
    'dyinp': (ylat[-1] - ylat[0])/(ylat.size-1),
}

sp.set_main_mesh(main_mesh)


# --------------------------------------
# SWAN parameters (sea level, jonswap gamma, ...)
input_params = {
    'set_level': 0,
    'set_convention': 'NAUTICAL',  # NAUTICAL, angle convention for wind/waves
    #'coords_projection': 'CCM',   # projection method: 'CCM' (default), 'QC'
    'coords_mode': 'CARTESIAN',    # coordinates system 'CARTESIAN' (default), 'SPHERICAL' 
    #'set_cdcap': 2.5*10**-3,

    #'boundw_jonswap': 3.3,
    #'boundw_period': 'MEAN',

    #'boundn_mode':'CLOSED',

    'wind_deltinp': '1 HR',
    'level_deltinp': '1 HR',

    'compute_deltc': '30 MIN',
    'output_deltt': '1 HR',         # MUST BE MULTIPLE OF 'compute_deltc'

    'physics':[
        #'WIND DRAG WU',
        #'GEN3 ST6 5.7E-7 8.0E-6 4.0 4.0 UP HWANG VECTAU TRUE10',
        #'QUAD iquad=8',
        #'WCAP',
        #'SETUP',  # not compatible with spherical coords
        #'TRIADS',
        #'DIFFRAC',
    ],

    'numerics':[
        'PROP BSBT',
    ],

    # BLOCK, compgrid output
    'output_variables': ['HSIGN','TPS','TM02','DIR','WIND'],
    #'output_time_ini_block': 24,

    # TABLE
    'output_points_x': [1500000, 2000000, 3000000, 4000000-50000],
    'output_points_y': [dy/2, dy/2, dy/2, dy/2],
    #'output_points_spec': True,
    'output_spec_deltt': '1 HR',
    'output_variables_points': ['HSIGN','TPS','TM02','DIR','WIND'],
}

sp.set_params(input_params)


# --------------------------------------
# SWAN case input: 2D Wind

# time array for case input
days = 2
hours = 24*days
date_ini = '2000-01-01 00:00'
time = pd.date_range(date_ini, periods=hours, freq='{0}min'.format(60))

# wind computational grid
mxc = main_mesh.cg['mxc']  # number mesh x
myc = main_mesh.cg['myc']  # number mesh y

x_wind = np.arange(main_mesh.cg['xpc'], main_mesh.cg['xlenc'], main_mesh.cg['dxinp'])
y_wind = np.arange(main_mesh.cg['ypc'], main_mesh.cg['ylenc'], main_mesh.cg['dyinp'])

# wind variable during time 
u_2d = np.zeros((myc, mxc, len(time)))
v_2d = np.zeros((myc, mxc, len(time)))

# customize 2D winds
u_2d[:, :50, :24] = 20

wind_2d = xr.Dataset(
    {
        'U10':(('y', 'x', 'time'), u_2d),
        'V10':(('y', 'x', 'time'), v_2d),
    },
    coords = {
        'y': y_wind,
        'x': x_wind,
        'time': time,
    }
)


# set case input
si = SwanInput_NONSTAT()

si.wind_mode = '2D'
si.wind_series = wind_2d


# --------------------------------------
# SWAN wrap NONSTAT (create case files, launch SWAN num. model, extract output)

sw = SwanWrap_NONSTAT(sp)

# build non-stationary cases
sw.build_cases([si])

# run SWAN
sw.run_cases()

# extract output from non-stationary cases
xds_out_main = sw.extract_output()
print('\noutput main mesh')
print(xds_out_main)

