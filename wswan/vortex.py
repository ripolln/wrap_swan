#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr

from .geo import geo_distance_azimuth, geo_distance_cartesian


def get_xy_labels(coords_mode):
    'returns labels for x,y axis. function of swan coordinatess mode'

    if coords_mode == 'SPHERICAL':
        lab_x, lab_x_long = 'lon', 'Longitude (º)'
        lab_y, lab_y_long = 'lat', 'Latitude (º)'

    elif coords_mode == 'CARTESIAN':
        lab_x, lab_x_long = 'X', 'X (m)'
        lab_y, lab_y_long = 'Y', 'Y (m)'

    return lab_x, lab_x_long, lab_y, lab_y_long

def geo_distance_meters(y_matrix, x_matrix, y_point, x_point, coords_mode):
    '''
    Returns distance between matrix and point (in meters)
    '''
    RE = 6378.135 * 1000            # Earth radius [m]

    if coords_mode == 'SPHERICAL':
        arcl, _ = geo_distance_azimuth(y_matrix, x_matrix, y_point, x_point)
        r = arcl * np.pi / 180.0 * RE  # to meteres

    if coords_mode == 'CARTESIAN':
        r = geo_distance_cartesian(y_matrix, x_matrix, y_point, x_point)

    return r

def vortex_model(storm_track, swan_mesh, coords_mode='SPHERICAL'):
    '''
    Uses winds vortex model to generate wind fields from storm track parameters

    Wind model code (from ADCIRC, transcribed by Antonio Espejo) and
    later slightly modified by Sara Ortega to include TCs at southern

    storm_track - (pandas.DataFrame)
    - obligatory fields:
        vfx
        vfy
        p0
        pn
        index
        vmax

    - optional fields:
        rmw  (optional)

    - for SPHERICAL coordinates
        lon / x
        lat / y

    - for CARTESIAN coordiantes:
        x
        y
        latitude

    swan_mesh.computational_grid
        mxc
        myc
        xpc
        ypc
        xlenc
        ylenc

    coords_mode - 'SHPERICAL' / 'CARTESIAN' swan project coordinates mode
    '''

    # parameters
    RE = 6378.135 * 1000            # Earth radius [m]
    beta = 0.9                      # conversion factor of wind speed
    rho_air = 1.15                  # air density
    w = 2 * np.pi / 86184.2         # Earth's rotation velocity (rad/s)
    pifac = np.arccos(-1) / 180     # pi/180
    one2ten = 0.8928                # conversion from 1-min to 10-min

    # wind variables
    storm_vfx  = storm_track.vfx.values[:]
    storm_vfy  = storm_track.vfy.values[:]
    storm_p0   = storm_track.p0.values[:]
    storm_pn   = storm_track.pn.values[:]
    times      = storm_track.index[:]
    storm_vmax = storm_track.vmax.values[:]

    # optional wind variables
    if 'rmw' in storm_track:
        storm_rmw  = storm_track.rmw.values[:]
    else:
        storm_rmw = [None] * len(storm_vfx)

    # coordinate system dependant variables 
    if coords_mode == 'SPHERICAL':
        storm_x  = storm_track.lon.values[:]
        storm_y  = storm_track.lat.values[:]
        storm_lat  = storm_track.lat.values[:]

    if coords_mode == 'CARTESIAN':
        storm_x    = storm_track.x.values[:]
        storm_y    = storm_track.y.values[:]
        storm_lat  = storm_track.latitude.values[:]

    # Correction when track is in south hemisphere for vortex generation 
    south_hemisphere = any (i < 0 for i in storm_lat)

    # swan mesh: computational grid (for generating vortex wind)
    mxc  = swan_mesh.cg['mxc']
    myc  = swan_mesh.cg['myc']
    xpc = swan_mesh.cg['xpc']
    ypc = swan_mesh.cg['ypc']
    xpc_xlenc = swan_mesh.cg['xpc'] + swan_mesh.cg['xlenc']
    ypc_ylenc = swan_mesh.cg['ypc'] + swan_mesh.cg['ylenc']

    # prepare meshgrid
    cg_lon = np.linspace(xpc, xpc_xlenc, mxc)
    cg_lat = np.linspace(ypc, ypc_ylenc, myc)
    mg_lon, mg_lat = np.meshgrid(cg_lon, cg_lat)

    # wind output holder
    hld_W = np.zeros((len(cg_lat), len(cg_lon), len(storm_p0)))
    hld_D = np.zeros((len(cg_lat), len(cg_lon), len(storm_p0)))

    # each time needs 2D (mesh) wind files (U,V)
    for c, (lo, la, la_orig, p0, pn, ut, vt, vmax, rmw) in enumerate(zip(
        storm_x, storm_y, storm_lat, storm_p0, storm_pn,
        storm_vfx, storm_vfy, storm_vmax, storm_rmw)):

        # generate vortex field when storm is given
        if all (np.isnan(i) for i in (lo, la, la_orig, p0, pn, ut, vt, vmax)) == False:

            # get distance and angle between points 
            r = geo_distance_meters(mg_lat, mg_lon, la, lo, coords_mode)

            # angle correction for southern hemisphere
            if south_hemisphere:
                thet = np.arctan2((mg_lat-la)*pifac, -(mg_lon-lo)*pifac)
            else:
                thet = np.arctan2((mg_lat-la)*pifac, (mg_lon-lo)*pifac)

            # ADCIRC model 
            CPD = (pn - p0) * 100    # central pressure deficit [Pa]
            if CPD < 100: CPD = 100  # limit central pressure deficit

            # Wind model 
            f = 2 * w * np.sin(abs(la_orig)*np.pi/180)  # Coriolis

            # Substract the translational storm speed from the observed maximum 
            # wind speed to avoid distortion in the Holland curve fit. 
            # The translational speed will be added back later
            vkt = vmax - np.sqrt(np.power(ut,2) + np.power(vt,2))  # [kt]

            # Convert wind speed from 10m altitude to wind speed at the top of 
            # the atmospheric boundary layer
            vgrad = vkt / beta  # [kt]
            v = vgrad
            vm = vgrad * 0.52  # [m/s]

            # TODO revisar 
            # optional rmw
            #if rmw == None:
            if True:

                # Knaff et al. (2016) - Radius of maximum wind (RMW)
                rm = 218.3784 - 1.2014*v + np.power(v/10.9844,2) - \
                        np.power(v/35.3052,3) - 145.509*np.cos(la_orig*pifac)  # nautical mile
                rm = rm * 1.852 * 1000   # from nautical mile to meters 

            else:
                rm = rmw

            rn = rm / r  # dimensionless

            # Holland B parameter with upper and lower limits
            B = rho_air * np.exp(1) * np.power(vm,2) / CPD
            if B > 2.5: B = 2.5
            elif B < 1: B = 1

            # Wind velocity at each node and time step   [m/s]
            vg = np.sqrt(np.power(rn,B) * np.exp(1-np.power(rn,B)) * \
                         np.power(vm,2) + np.power(r,2)*np.power(f,2)/4) - r*f/2

            # Determine translation speed that should be added to final storm  
            # wind speed. This is tapered to zero as the storm wind tapers to 
            # zero toward the eye of the storm and at long distances from the storm
            vtae = (abs(vg) / vgrad) * ut    # [m/s]
            vtan = (abs(vg) / vgrad) * vt

            # Find the velocity components and convert from wind at the top of the 
            # atmospheric boundary layer to wind at 10m elevation
            hemisphere_sign = 1 if south_hemisphere else -1
            ve = hemisphere_sign * vg * beta * np.sin(thet)  # [m/s]
            vn = vg * beta * np.cos(thet)

            # Convert from 1 minute averaged winds to 10 minute averaged winds
            ve = ve * one2ten    # [m/s]
            vn = vn * one2ten

            # Add the storm translation speed
            vfe = ve + vtae      # [m/s]
            vfn = vn + vtan

            # wind module
            W = np.sqrt(np.power(vfe,2) + np.power(vfn,2))  # [m/s]

            # Surface pressure field
            pr = p0 + (pn-p0) * np.exp(- np.power(rn,B))      # [mbar]
            py, px = np.gradient(pr)
            ang = np.arctan2(py, px) + np.sign(la_orig) * np.pi/2.0

            # hold wind data (m/s)
            hld_W[:,:,c] = W
            hld_D[:,:,c] =  270 - np.rad2deg(ang)  # direction (º clock. rel. north)

        else:
            # hold wind data (m/s)
            hld_W[:,:,c] = 0
            hld_D[:,:,c] = 0  # direction (º clock. rel. north)

    # spatial axis labels
    lab_x, lab_x_long, lab_y, lab_y_long = get_xy_labels(coords_mode)

    # generate vortex dataset 
    xds_vortex = xr.Dataset(
        {
            'W':   ((lab_y, lab_x, 'time'), hld_W, {'units':'m/s'}),
            'Dir': ((lab_y, lab_x, 'time'), hld_D, {'units':'º'})
        },
        coords={
            lab_y : cg_lat,
            lab_x : cg_lon,
            'time' : times,
        }
    )
    xds_vortex.attrs['xlabel'] = lab_x_long
    xds_vortex.attrs['ylabel'] = lab_y_long

    return xds_vortex

