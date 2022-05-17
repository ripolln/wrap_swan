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
from wswan.wrap import SwanProject, SwanMesh, SwanWrap_NONSTAT


# --------------------------------------
# data
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
p_demo = op.join(p_data, 'demo')

# test data: csiro point 
p_waves_demo = op.join(p_demo, 'waves_csiro_demo.nc')


# --------------------------------------
# SWAN case input: waves_event 

# non-stationary case requires a wave_event (dim: time). variables:
# time
# waves: hs, t02, dir, spr
# water level: level, tide     (not in csiro?)

xds_waves = xr.open_dataset(p_waves_demo)
xds_waves = xds_waves.squeeze()   # remove lon,lat dim (len=1)
waves = xds_waves.to_dataframe()  # xarray --> pandas

# now we generate the wave event 
vs = ['hs', 't02', 'dir', 'spr']
we = waves['2000-01-02 00:00':'2000-01-02 03:00'][vs]
we['level'] = 0 # no water level data 
we['tide'] = 0 # no tide data 
we.rename(columns={'t02': 'per'}, inplace=True)  # rename for swan

print('\ninput wave event')
print(we)


# --------------------------------------
# SWAN project (config bathymetry, parameters, computational grid)

p_proj = op.join(p_data, 'projects')  # swan projects main directory
n_proj = '02_nonstat_wind2d'          # project name

sp = SwanProject(p_proj, n_proj)


# --------------------------------------
# SWAN main mesh
main_mesh = SwanMesh()

# depth grid description (input bathymetry grid)
main_mesh.dg = {
    'xpc': 0,       # x origin
    'ypc': 0,       # y origin
    'alpc': 0,      # x-axis direction 
    'xlenc': 400,   # grid length in x
    'ylenc': 400,  # grid length in y
    'mxc': 1,       # number mesh x
    'myc': 1,       # number mesh y
    'dxinp': 400,   # size mesh x
    'dyinp': 400,  # size mesh y
}

# depth value
main_mesh.depth = np.ones((2,2)) * 155

# computational grid description
main_mesh.cg = {
    'xpc': 0,
    'ypc': 0,
    'alpc': 0,
    'xlenc': 400,
    'ylenc': 400,
    'mxc': 40,
    'myc': 20,
    'dxinp': 10,
    'dyinp': 20,
}

sp.set_main_mesh(main_mesh)


# --------------------------------------
# SWAN parameters (sea level, jonswap gamma, ...)
input_params = {
    'set_level': 4,
    'set_convention': 'NAUTICAL',

    'boundw_jonswap': 1.9,
    'boundw_period': 'MEAN',

    'boundn_mode': 'CLOSED',

    'wind_deltinp': '1 HR',
    'level_deltinp': '1 HR',

    'compute_deltc': '5 MIN',
    'output_deltt': '30 MIN',

    'output_points_x': [50.5, 70, 120],
    'output_points_y': [60, 120, 160],

    'output_variables': [
        'HSIGN', 'DIR', 'PDIR', 'TM02',
        'TPS', 'RTP', 'FSPR', 'DSPR',
        'WIND', 'DEPTH', 'WATLEV', 'WIND',
        #'PTHSIGN', 'PTDIR', 'PTRTP', 'PTDSPR',
        'PTWFRAC', 'PTWLEN', 'PTSTEEP',
    ],

    'physics':[
        'WIND DRAG WU',
        'GEN3 ST6 5.7E-7 8.0E-6 4.0 4.0 UP HWANG VECTAU TRUE10',
        'QUAD iquad=8',
        #'OFF QUAD',
        'WCAP',
        #'SETUP',  # not compatible with spherical coords
        'TRIADS',
        'DIFFRAC',
    ],

    'numerics':[
        'PROP BSBT',
    ]
}
sp.set_params(input_params)

# --------------------------------------
# generate custom 2D wind for demo

# wind computational grid
mxc = main_mesh.cg['mxc']  # number mesh x
myc = main_mesh.cg['myc']  # number mesh y

x_wind = np.arange(main_mesh.cg['xpc'], main_mesh.cg['xlenc'], main_mesh.cg['dxinp'])
y_wind = np.arange(main_mesh.cg['ypc'], main_mesh.cg['ylenc'], main_mesh.cg['dyinp'])

t_wind = we.index.values[:]

# wind variable during time 
aux = np.ones((mxc, myc))

u_2d = np.zeros((mxc, myc, len(t_wind)))
v_2d = np.zeros((mxc, myc, len(t_wind)))

for i in range(len(t_wind)):

    u_2d[:,:,i] = aux * 2 * i
    v_2d[:,:,i] = aux * 4 * i


# wind 2D dataset
wind_2d = xr.Dataset(
    {
        'U10':(('x', 'y', 'time'), u_2d),
        'V10':(('x', 'y', 'time'), v_2d),
    },
    coords = {
        'x': x_wind,
        'y': y_wind,
        'time': t_wind,
    }
)


# --------------------------------------
# SWAN wrap NONSTAT (create case files, launch SWAN num. model, extract output)

sw = SwanWrap_NONSTAT(sp)

# build non-stationary cases from wave_events list
sw.build_cases(
    [we], wind_2d_list=[wind_2d],
    make_winds=True, make_levels=True,
)  # test one event

# run SWAN
sw.run_cases()

# extract output from non-stationary cases
xds_out_main = sw.extract_output()
print('\noutput main mesh')
print(xds_out_main)

