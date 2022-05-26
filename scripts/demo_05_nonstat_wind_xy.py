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


# --------------------------------------
# data
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
p_demo = op.join(p_data, 'demo')


# --------------------------------------
# SWAN case input: waves event + winds + level

# wind: U10, V10 (units_ m/s)
wind_series = pd.DataFrame(index=pd.date_range("20220101", periods=1*24, freq='H'))

wind_series['U10'] = 20
wind_series['V10'] = 0

print('\ninput wind series')
print(wind_series)


# --------------------------------------

# set case input
si = SwanInput_NONSTAT()

si.wind_mode = 'uniform'
si.wind_series = wind_series


# --------------------------------------
# SWAN project (config bathymetry, parameters, computational grid)

p_proj = op.join(p_data, 'projects')  # swan projects main directory
n_proj = '05_winds_xy'                # project name

sp = SwanProject(p_proj, n_proj)


# --------------------------------------
# SWAN main mesh
main_mesh = SwanMesh()

# depth grid description (input bathymetry grid)
main_mesh.dg = {
    'xpc': 0,          # x origin
    'ypc': 0,          # y origin
    'alpc': 0,         # x-axis direction 
    'xlenc': 4000000,  # grid length in x
    'ylenc': 500000,   # grid length in y
    'mxc': 1,          # number mesh x
    'myc': 1,          # number mesh y
    'dxinp': 4000000,  # size mesh x
    'dyinp': 500000,   # size mesh y
}

# depth value
main_mesh.depth = np.ones((2,2)) * 200

# computational grid description
main_mesh.cg = {
    'xpc': 0,
    'ypc': 0,
    'alpc': 0,
    'xlenc': 4000000,
    'ylenc': 500000,
    'mxc': 200,
    'myc': 25,
    'dxinp': 20000,
    'dyinp': 20000,
}

sp.set_main_mesh(main_mesh)


# --------------------------------------
# SWAN parameters (sea level, jonswap gamma, ...)
input_params = {
    'set_level': 0,
    'set_convention': 'NAUTICAL',

    'coords_mode': 'CARTESIAN',

    #'boundw_jonswap': 1.9,
    #'boundw_period': 'MEAN',

    #'boundn_mode': 'CLOSED',

    'wind_deltinp': '1 HR',
    'level_deltinp': '1 HR',

    'compute_deltc': '30 MIN',
    'output_deltt': '1 HR',

    'output_points_x': [1500000, 2000000, 3000000, 4000000-50000],
    'output_points_y': [250000, 250000, 250000, 250000],

    'output_variables': ['HSIGN', 'DIR', 'TM02', 'TPS', 'WIND'],
    'output_variables_points': ['HSIGN','TPS','TM02','DIR','WIND'],

    'physics':[
        #'WIND DRAG WU',
        #'GEN3 ST6 5.7E-7 8.0E-6 4.0 4.0 UP HWANG VECTAU TRUE10',
        #'QUAD iquad=8',
        #'OFF QUAD',
        #'WCAP',
        #'SETUP',  # not compatible with spherical coords
        #'TRIADS',
        #'DIFFRAC',
    ],

    'numerics':[
        'PROP BSBT',
    ]
}
sp.set_params(input_params)


# --------------------------------------
# SWAN wrap NONSTAT (create case files, launch SWAN num. model, extract output)

sw = SwanWrap_NONSTAT(sp)

# build non-stationary cases from SwanInput list
sw.build_cases([si])  # test one event

# run SWAN
sw.run_cases()

# extract output from non-stationary cases
xds_out_main = sw.extract_output()
print('\noutput main mesh')
print(xds_out_main)


# Plot Hsig 
from wswan.plots.nonstationary import plot_case_output
import matplotlib.pyplot as plt

plot_case_output(sw, t_num=23)
plt.show()

