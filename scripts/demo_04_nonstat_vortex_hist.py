#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import sys
import os
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# dev
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# swan wrap module 
from wswan.storms import historic_track_preprocessing, historic_track_interpolation
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

# shoreline (from .nc file)
p_shore = op.join(p_demo, 'majuro', 'shore_majuro.npy')

np_shore = np.load(p_shore)

# sign convention [0º,360º]
lon_shore = np_shore[:,0]
lon_shore[lon_shore<0] = lon_shore[lon_shore<0] + 360
np_shore[:,0] = lon_shore


# historic storm
storm = xr.open_dataset(op.join(p_demo, 'majuro', 'storm_ibtracs_paka.nc'))

# target coordinates
target = 'Kwajalein'
x0, y0 = 167.73, 8.72
if x0 < 0:  x0 = x0 + 360   # sign convention [0º,360º]



# --------------------------------------
# SWAN project (config bathymetry, parameters, computational grid)

p_proj = op.join(p_data, 'projects')  # swan projects main directory
n_proj = '04_vx_hist'                 # project name

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
# SWAN nested meshes


# NEST 1
mesh_nest1 = SwanMesh()

# depth grid description (input bathymetry grid)
# grid resolution of 5km (=0.0453º)
res1 = 0.04533
mesh_nest1.dg = main_mesh.dg

# depth value (from file)
mesh_nest1.depth = main_mesh.depth

# computational grid description
mesh_nest1.cg = {
    'xpc': 168.5,
    'ypc': 5.5,
    'alpc': 0,
    'xlenc': 5.5,
    'ylenc': 3.5,
    'mxc': int(round(5.5/res1)),
    'myc': int(round(3.5/res1)),
    'dxinp': 5.5/int(round(5.5/res1)),
    'dyinp': 3.5/int(round(3.5/res1)),
}


# NEST 2
mesh_nest2 = SwanMesh()

# depth grid description (input bathymetry grid)
# grid resolution of 1km (=0.009º)
res2 = 0.009
mesh_nest2.dg = main_mesh.dg

# depth value (from file)
mesh_nest2.depth = main_mesh.depth

# computational grid description
mesh_nest2.cg = {
    'xpc': 170.9,
    'ypc': 6.8,
    'alpc': 0,
    'xlenc': 1.2,
    'ylenc': 0.7,
    'mxc': int(round(1.2/res2)),
    'myc': int(round(0.7/res2)),
    'dxinp': 1.2/int(round(1.2/res2)),
    'dyinp': 0.7/int(round(0.7/res2)),
}


# set project nested mesh list
sp.set_nested_mesh_list([mesh_nest1, mesh_nest2])


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

    'boundn_mode': 'CLOSED',

    'wind_deltinp': '30 MIN',
    'level_deltinp': '1 HR',

    'compute_deltc': '30 MIN',
    'output_deltt': '30 MIN',

    'output_points_x': [172.5, 172.5, 171],
    'output_points_y': [8.5, 9.6, 7.4],

    'physics':[
        'WIND DRAG WU',
        'GEN3 ST6 5.7E-7 8.0E-6 4.0 4.0 UP HWANG VECTAU TRUE10',
        'QUAD iquad=8',
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
# SWAN case input: storm_track (from historic storm)

# variables names for v3.10 Ibtracs
d_vns = {
    'longitude':'lon_wmo',
    'latitude':'lat_wmo',
    'time': 'time_wmo',
    'pressure':'pres_wmo',
    'maxwinds':'wind_wmo',
}

# preprocess storm variables
st_time, ylat_tc, ylon_tc, ycpres, ywind, ts, categ, vmean = historic_track_preprocessing(storm, d_vns)

# generate interpolated storm track  
dt_interp = 30  # minutes
st, time_input = historic_track_interpolation(
    st_time, ylon_tc, ylat_tc, ycpres, ywind,
    y0, x0, lat[0], lon[0], lat[-1], lon[-1],
    ts, dt_interp, wind=ywind,
    great_circle=True)

print('\ninput storm track')
print(st)


# set case input
si = SwanInput_NONSTAT()

si.wind_mode = 'storm'
si.wind_series = st


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

xds_out_nest1 = sw.extract_output(mesh=sp.mesh_nested_list[0])
print('\noutput nest1 mesh')
print(xds_out_nest1)

xds_out_nest2 = sw.extract_output(mesh=sp.mesh_nested_list[1])
print('\noutput nest2 mesh')
print(xds_out_nest2)

# extract point output from non-stationary cases
xds_out_pts = sw.extract_output_points()


