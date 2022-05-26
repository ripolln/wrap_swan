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

# test data: csiro point 
p_waves_demo = op.join(p_demo, 'waves_csiro_demo.nc')


# --------------------------------------
# SWAN case input:

# waves event: hs, t02, dir, spr
# water level: level, tide
# wind: U10, V10

# load csiro point dataset
xds_csiro = xr.open_dataset(p_waves_demo)
xds_csiro = xds_csiro.squeeze()         # remove lon,lat dim (len=1)
point_csiro = xds_csiro.to_dataframe()  # xarray --> pandas

# wave event 
vs = ['hs', 't02', 'dir', 'spr']
waves_event = point_csiro['2000-01-02 00:00':'2000-01-02 03:00'][vs]
waves_event.rename(columns={'t02': 'per'}, inplace=True)  # rename for swan

print('\ninput waves event')
print(waves_event)

# water level
water_level = pd.DataFrame(index=waves_event.index)
water_level['level'] = 0
water_level['tide'] = 0

print('\ninput water level and tide')
print(water_level)


# wind: U10, V10 (units_ m/s)
vs = ['U10', 'V10']
wind_series = point_csiro['2000-01-02 00:00':'2000-01-02 03:00'][vs]

print('\ninput wind series')
print(wind_series)


# --------------------------------------

# set case input
si = SwanInput_NONSTAT()

si.waves_activate = True
si.waves_series = waves_event

si.level_activate = True
si.level_series = water_level

si.wind_mode = 'uniform'
si.wind_series = wind_series



# --------------------------------------
# SWAN project (config bathymetry, parameters, computational grid)

p_proj = op.join(p_data, 'projects')  # swan projects main directory
n_proj = '02_nonstat'                 # project name

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
# SWAN nest1 mesh
mesh_nest1 = SwanMesh()

# depth grid description
mesh_nest1.dg = {
    'xpc': 50,
    'ypc': 100,
    'alpc': 0,
    'xlenc': 80,
    'ylenc': 100,
    'mxc': 8,
    'myc': 10,
    'dxinp': 10,
    'dyinp': 10,
}

# depth value
mesh_nest1.depth = np.ones((10,8)) * 155

# computational grid description
mesh_nest1.cg = {
    'xpc': 50,
    'ypc': 100,
    'alpc': 0,
    'xlenc': 80,
    'ylenc': 100,
    'mxc': 8,
    'myc': 10,
    'dxinp': 10,
    'dyinp': 10,
}

sp.set_nested_mesh_list([mesh_nest1])

# --------------------------------------
# SWAN parameters (sea level, jonswap gamma, ...)
input_params = {
    'set_level': 0,
    'set_convention': 'CARTESIAN',

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
        'WIND', 'DEPTH', 'WATLEV',
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

# extract output from nest1 mesh 
xds_out_nest1 = sw.extract_output(mesh=sp.mesh_nested_list[0])
print('\noutput nest1 mesh')
print(xds_out_nest1)

