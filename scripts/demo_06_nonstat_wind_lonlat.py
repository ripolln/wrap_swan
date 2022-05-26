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

# bathymetry (from .nc file)
p_bathy = op.join(p_demo, 'majuro', 'depth_majuro.nc')

xds_bathy = xr.open_dataset(p_bathy)

# sign convention [0º,360º]
xds_lon = xds_bathy.lon.values
xds_lon[xds_lon<0] = xds_lon[xds_lon<0] + 360
xds_bathy.lon.values[:] = xds_lon

lon = xds_bathy.lon.values[:]
lat = xds_bathy.lat.values[:]

depth = xds_bathy.elevation.values[:] * -1  # elevation to depth 


# --------------------------------------
# SWAN case input: winds 

# wind: U10, V10 (units_ m/s)
wind_series = pd.DataFrame(index=pd.date_range("20220101", periods=1*24, freq='H'))
wind_series['U10'] = 20
wind_series['V10'] = 0

print('\ninput wind series')
print(wind_series)

# set case input
si = SwanInput_NONSTAT()

si.wind_mode = 'uniform'
si.wind_series = wind_series


# --------------------------------------
# SWAN project (config bathymetry, parameters, computational grid)

p_proj = op.join(p_data, 'projects')  # swan projects main directory
n_proj = '06_winds_lonlat'            # project name

sp = SwanProject(p_proj, n_proj)


# --------------------------------------
# SWAN main mesh
main_mesh = SwanMesh()

# depth grid description (input bathymetry grid)
# grid resolution of 15km (=0.136º)
res = 0.136
main_mesh.dg = {
    'xpc': lon[0],                             # x origin
    'ypc': lat[0],                             # y origin
    'alpc': 0,                                 # x-axis direction 
    'xlenc': lon[-1]-lon[0],                   # grid length in x
    'ylenc': lat[-1]-lat[0],                   # grid length in y
    'mxc': depth.shape[1]-1,                   # number mesh x
    'myc': depth.shape[0]-1,                   # number mesh y
    'dxinp': (lon[-1]-lon[0])/depth.shape[1],  # size mesh x
    'dyinp': (lat[-1]-lat[0])/depth.shape[0],  # size mesh y
}

# depth value (from file)
main_mesh.depth = depth

# computational grid description
main_mesh.cg = {
    'xpc': 163.5,
    'ypc': 0.5,
    'alpc': 0,
    'xlenc': 15,
    'ylenc': 13,
    'mxc': int(round(15/res)),    # grid resolution of 15km (=0.136º)
    'myc': int(round(13/res)),
    'dxinp': 15/int(round(15/res)),
    'dyinp': 13/int(round(13/res)),
}

sp.set_main_mesh(main_mesh)


# --------------------------------------
# SWAN parameters (sea level, jonswap gamma)
input_params = {
    'set_level': 0,
    'set_convention': 'NAUTICAL',
    'set_cdcap': 2.5*10**-3,

    'coords_mode': 'SPHERICAL',
    'coords_projection': 'CCM',

    'boundw_jonswap': 3.3,
    'boundw_period': 'MEAN',

    #'boundn_mode': 'CLOSED',

    'wind_deltinp': '1 HR',
    'level_deltinp': '1 HR',

    'compute_deltc': '30 MIN',
    'output_deltt': '30 MIN',

    'output_points_x': [172.5, 172.5, 171],
    'output_points_y': [8.5, 9.6, 7.4],

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
    ]
}
sp.set_params(input_params)


# --------------------------------------
# SWAN wrap NONSTAT (create case files, launch SWAN num. model, extract output)

sw = SwanWrap_NONSTAT(sp)

# build non-stationary cases from wave_events list and storm_tracks list
sw.build_cases([si])

# run SWAN
sw.run_cases()

# extract grid output from non-stationary cases
xds_out_main = sw.extract_output()
print('\noutput main mesh')
print(xds_out_main)


# Plot Hsig 
#from wswan.plots.nonstationary import plot_case_output
#import matplotlib.pyplot as plt

#plot_case_output(sw, t_num=23)
#plt.show()

