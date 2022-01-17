#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import timedelta

from .geo import shoot, gc_distance


# STORM TRACK LIBRARY

def get_category(ycpres):
    'Defines storm category according to minimum pressure center'

    categ = []
    for i in range(len(ycpres)):
        if (ycpres[i] == 0) or (np.isnan(ycpres[i])):
            categ.append(6)
        elif ycpres[i] < 920:  categ.append(5)
        elif ycpres[i] < 944:  categ.append(4)
        elif ycpres[i] < 964:  categ.append(3)
        elif ycpres[i] < 979:  categ.append(2)
        elif ycpres[i] < 1000: categ.append(1)
        elif ycpres[i] >= 1000: categ.append(0)

    return categ

def historic_track_preprocessing(xds, d_vns):
    '''
    Historic track is preprocessed, by removing NaN data, apply longitude
    convention [0º-360º], change time format and define storm category

    xds:   historic track dataset (storm dimension)
    d_vns: dictionary for longitude, latitude, time, pressure and wind varnames

    returns variables:
           time, lat, lon, pressure, wind, timestep, category, mean translational speed
    '''

    # get names of vars
    nm_lon = d_vns['longitude']
    nm_lat = d_vns['latitude']
    nm_prs = d_vns['pressure']
    nm_tim = d_vns['time']
    nm_win = d_vns['maxwinds']

    # get var time
    ytime = xds[nm_tim].values         # datetime64

    # remove NaTs (time)
    ycpres = xds[nm_prs].values[~np.isnat(ytime)]     # minimum pressure
    ylat_tc = xds[nm_lat].values[~np.isnat(ytime)]    # latitude
    ylon_tc = xds[nm_lon].values[~np.isnat(ytime)]    # longitude
    ywind = xds[nm_win].values[~np.isnat(ytime)]      # wind speed [kt]
    ytime = ytime[~np.isnat(ytime)]

    # remove NaNs (pressure)
    ylat_tc = ylat_tc[~np.isnan(ycpres)]
    ylon_tc = ylon_tc[~np.isnan(ycpres)]
    ywind = ywind[~np.isnan(ycpres)]
    ytime = ytime[~np.isnan(ycpres)]
    ycpres = ycpres[~np.isnan(ycpres)]

    # longitude convention: [0º,360º]
    ylon_tc[ylon_tc<0] = ylon_tc[ylon_tc<0] + 360

    # round dates to hour
    round_to = 3600
    st_time = []
    for i in range(len(ytime)):
        dt = ytime[i].astype('datetime64[s]').tolist()
        seconds = (dt - dt.min).seconds
        rounding = (seconds+round_to/2) // round_to * round_to
        out = dt + timedelta(0, rounding-seconds, -dt.microsecond)
        st_time.append(out)
    st_time = np.asarray(st_time)

    # storm coordinates timestep [hours]
    ts = (st_time[1:] - st_time[:-1])
    ts = [ts[i].total_seconds() / 3600 for i in range(ts.size)]

    # storm coordinates category
    categ = get_category(ycpres)

    # calculate Vmean
    RE = 6378.135   # earth radius [km]
    vmean = []
    for i in range(0, len(st_time)-1):

        # consecutive storm coordinates
        lon1, lon2 = ylon_tc[i], ylon_tc[i+1]
        lat1, lat2 = ylat_tc[i], ylat_tc[i+1]

        # translation speed 
        arcl_h, gamma_h = gc_distance(lat2, lon2, lat1, lon1)
        r = arcl_h * np.pi / 180.0 * RE     # distance between consecutive coordinates (km)
        vmean.append(r / ts[i] / 1.852)     # translation speed (km/h to kt) 

    # mean value
    vmean = np.mean(vmean)  # [kt]

    return st_time, ylat_tc, ylon_tc, ycpres, ywind, ts, categ, vmean

def ibtrac_basin_fitting(x0, y0):
    '''
    Assigns cubic polynomial fitting curve coefficient for each basin of
    historical TCs data (IBTrACS)
    '''

    # determination of the location basin 
    if y0 < 0:                  basin = 5
    elif (y0 > 0) & (x0 > 0):   basin = 3
    else:                       print('Basin not defined')

    # cubic polynomial fitting curve for Ibtracs and each basin
    # TODO: obtain all basin fitting coefficients

    if basin == 3:      # West Pacific
        p1 = -7.77328602747578e-06
        p2 = 0.0190830514629838
        p3 = -15.9630945598490
        p4 = 4687.76462404360

    elif basin == 5:    # South Pacific
        p1 = -4.70481986864773e-05
        p2 = 0.131052968357409
        p3 = -122.487981649828
        p4 = 38509.7575283218

    return p1, p2, p3, p4

def historic_track_interpolation(st_time, ylon_tc, ylat_tc, ycpres, ywind, y0, x0,
                                 lat00, lon00, lat01, lon01, ts, dt_comp,
                                 wind=None, great_circle=False, fit=False, 
                                 interpolation=True, mode='first'):
    '''
    Calculates storm track variables from storm track parameters and interpolates
    track points in between historical data (for "dt_comp" time step)

    st_time                    - storm dates
    ylon_tc, ylat_tc           - storm coordinates (longitude, latitude)
    ycpres, ywind              - storm minimum pressure, maximum winds
    x0, y0                     - target coordinates (longitude, latitude)
    lat0, lon0, lat1, lon1     - bound limits for numerical domain (longitude, latitude)
    ts                         - storm coordinates time step [hours]
    dt_comp                    - computation time step [minutes]

    wind                       - historical ywind can be assigned or 'None' for vmax empirical estimate
    great_circle               - True for distances over great circle 
    fit                        - True for fitting empirical vmax when ywind=0 (start of storm)
    imterpolation              - True for storm variables interpolation (historic storms)
                                 False for mean storm variables (storm segments constant)
    mode ('first','mean')      - when interpolation is activated, chooses value for constant segments

    returns:  dataframe with interpolated storm coordinate variables 
    '''

    RE = 6378.135   # earth radius [km]

    # cubic polynomial fitting coefficients for IBTrACS basins Pmin-Vmax relationship
    p1, p2, p3, p4 = ibtrac_basin_fitting(x0, y0)

    # storm variables
    time_storm = list(st_time)  # datetime format
    pmin = list(ycpres)
    lat = list(ylat_tc)
    lon = list(ylon_tc)
    if wind.any() != None:  # wind variable, no data is filled with IBTrACS fitting coefficients
        mwind = wind
        if fit:
            wind_fitting = p1 * np.power(pmin,3) + p2 * np.power(pmin,2) + p3 * np.power(pmin,1) + p4
            pos = np.where(mwind==0)
            mwind[pos] = wind_fitting[pos]

    # number of time steps between consecutive interpolated storm coordinates in order
    # to  match SWAN computational time step
    ts_h = ts                               # hours
    ts = np.asarray(ts) * 60 / dt_comp      # number of intervals

    # initialize
    move, vmean, pn, p0, lon_t, lat_t, vmax = [], [], [], [], [], [], []
    vu, vy = [], []
    time_input = np.empty((0,),dtype='datetime64[ns]')

    for i in range(0, len(time_storm)-1):
        # time array for SWAN input
        date_ini = time_storm[i]
        time_input0 = pd.date_range(
                date_ini, periods=int(ts[i]), freq='{0}MIN'.format(dt_comp))
        time_input = np.append(np.array(time_input), np.array(time_input0))

        # consecutive storm coordinates
        lon1, lon2 = lon[i], lon[i+1]
        lat1, lat2 = lat[i], lat[i+1]

        # translation speed 
        arcl_h, gamma_h = gc_distance(lat2, lon2, lat1, lon1)
        r = arcl_h * np.pi / 180.0 * RE     # distance between consecutive storm coordinates [km]
        dx = r / ts[i]                      # distance during time step
        tx = ts_h[i] / ts[i]                # time period during time step
        vx = float(dx) / tx / 3.6           # translation speed [km to m/s]
        vx = vx /0.52                       # translation speed [m/s to kt]

        for j in range(int(ts[i])):
            # append storm track parameters
            move.append(gamma_h)
            vmean.append(vx)
            vu.append(vx * np.sin((gamma_h+180)*np.pi/180))
            vy.append(vx * np.cos((gamma_h+180)*np.pi/180))
            pn.append(1013)

            # append pmin, wind with/without interpolation along the storm track
            if interpolation:       p0.append(pmin[i] + j* (pmin[i+1]-pmin[i])/ts[i])
            if not interpolation:   
                if mode=='mean':    p0.append(np.mean((pmin[i], pmin[i+1])))
                elif mode=='first': p0.append(pmin[i])

            if wind.any() != None:
                if interpolation:   vmax.append(mwind[i] + j* (mwind[i+1]-mwind[i])/ts[i])    #[kt]
                if not interpolation:   
                    if mode=='mean': vmax.append(np.mean((mwind[i], mwind[i+1])))   
                    if mode=='first':vmax.append(mwind[i])     #[kt]

            # calculate timestep lon, lat
            if not great_circle:
                lon_h = lon1 - (dx*180/(RE*np.pi)) * np.sin(gamma_h*np.pi/180) * j
                lat_h = lat1 - (dx*180/(RE*np.pi)) * np.cos(gamma_h*np.pi/180) * j
            else:
                xt, yt = [], []
                glon, glat, baz = shoot(lon1, lat1, gamma_h + 180, float(dx) * j)
                xt = np.append(xt,glon)
                yt = np.append(yt,glat)
                lon_h = xt
                lat_h = yt
            lon_t.append(lon_h)
            lat_t.append(lat_h)

    # to array
    move = np.array(move)
    vmean = np.array(vmean)
    vu = np.array(vu)
    vy = np.array(vy)
    p0 = np.array(p0)
    vmax = np.array(vmax)
    lon_t = np.array(lon_t)
    lat_t = np.array(lat_t)

    # longitude convention [0º,360º]
    lon_t[lon_t<0]= lon_t[lon_t<0] + 360

    # select interpolation coordinates within the target domain area
    loc = []
    for i, (lo,la) in enumerate(zip(lon_t, lat_t)):
        if (lo<=lon01) & (lo>=lon00) & (la<=lat01) & (la>=lat00):
            loc.append(i)

    # storm track (pd.DataFrame)
    st = pd.DataFrame(index=time_input[loc],
                      columns=['move','vf','vfx','vfy','pn','p0','lon','lat','vmax'])

    st['move'] = move[loc]      # gamma, forward direction
    st['vf'] = vmean[loc]       # translational speed [kt]
    st['vfx'] = vu[loc]         # x-component
    st['vfy'] = vy[loc]         # y-component
    st['pn'] = 1013             # average pressure at the surface [mbar]
    st['p0'] = p0[loc]          # minimum central pressure [mbar]
    st['lon'] = lon_t[loc]      # longitude coordinate
    st['lat'] = lat_t[loc]      # latitude coordinate
    # maximum wind speed (if no value is given it is assigned the empirical Pmin-Vmax basin-fitting)
    if wind.any() != None:  st['vmax'] = vmax[loc]  # [kt]
    else:                   st['vmax'] = p1 * np.power(p0[loc],3) + p2 * np.power(p0[loc],2) + \
                                        p3 * np.power(p0[loc],1) + p4   # [kt]

    # add some metadata
    # TODO: move to st.attrs (this metada gets lost with any operation with st)
    st.x0 = x0
    st.y0 = y0
    st.R = 4

    return st, time_input[loc]

def entrance_coords(delta, gamma, x0, y0, R, lon0, lon1, lat0, lat1):
    '''
    Calculates storm track initial coordinates

    delta, gamma               - storm track parameters
    x0, y0                     - site coordinates (longitude, latitude)
    R                          - radius (º)
    lon0, lon1, lat0, lat1     - computational coordinates (outer grid)
    '''

    # enter point in the radius
    xc = x0 + R * np.sin(delta * np.pi/180)
    yc = y0 + R * np.cos(delta * np.pi/180)

    # calculate angles that determine the storm boundary entrance  [degrees]
    ang_1 = np.arctan((lon1-xc)/(lat1-yc)) *180/np.pi       # upper right corner
    ang_2 = np.arctan((lon1-xc)/(lat0-yc)) *180/np.pi +180  # lower right
    ang_3 = np.arctan((lon0-xc)/(lat0-yc)) *180/np.pi +180  # lower left
    ang_4 = np.arctan((lon0-xc)/(lat1-yc)) *180/np.pi +360  # upper left

    if (gamma > ang_1) & (gamma < ang_2):
        x1 = lon1
        d = (x1 - xc) / np.sin(gamma * np.pi/180)
        y1 = yc + d * np.cos(gamma * np.pi/180)

    elif (gamma > ang_2) & (gamma < ang_3):
        y1 = lat0
        d = (y1 - yc) / np.cos(gamma * np.pi/180)
        x1 = xc + d * np.sin(gamma * np.pi/180)

    elif (gamma > ang_3) & (gamma < ang_4):
        x1 = lon0
        d = (x1 - xc) / np.sin(gamma * np.pi/180)
        y1 = yc + d * np.cos(gamma * np.pi/180)

    elif (gamma > ang_4) | (gamma < ang_1):
        y1 = lat1
        d = (y1 - yc) / np.cos(gamma * np.pi/180)
        x1 = xc + d * np.sin(gamma * np.pi/180)

    return x1, y1

def track_site_parameters(step, pmin, vmean, delta, gamma,
                          x0, y0, lon0, lon1, lat0, lat1, R, date_ini):
    '''
    Calculates storm track variables from storm track parameters within the study area
    (uses great circle)

    step                       - computational time step (minutes)
    pmin, vmean, delta, gamma  - storm track parameters   (NOTE: vmean in [kt])
    x0, y0                     - site coordinates (longitude, latitude)
    lon0, lon1, lat0, lat1     - enter point in computational grid
    R                          - radius (º)
    date_ini                   - initial date 'yyyy-mm-dd HH:SS'
    great_circle               - default option
    '''

    # cubic polynomial fitting coefficients for IBTrACS basins Pmin-Vmax relationship
    p1, p2, p3, p4 = ibtrac_basin_fitting(x0, y0)

    # storm entrance coordinates at the domain boundary
    x1, y1 = entrance_coords(delta, gamma, x0, y0, R, lon0, lon1, lat0, lat1)

    # calculate computation storm coordinates
    xt, yt = [x1], [y1]
    i = 1
    glon, glat, baz = shoot(x1, y1, gamma+180, vmean*1.852 * i*step/60)  # velocity in [km/h]
    if glon < 0: glon += 360
    while (glon < lon1) & (glon > lon0) & (glat < lat1) & (glat > lat0):
        xt.append(glon)
        yt.append(glat)
        i += 1
        glon, glat, baz = shoot(x1, y1, gamma+180, vmean*1.852 * i*step/60)  # velocity in [km/h]
        if glon < 0: glon += 360
    frec = len(xt)

    # time array for SWAN input
    time_input = pd.date_range(date_ini, periods=frec, freq='{0}min'.format(step))

    # storm track (pd.DataFrame)
    st = pd.DataFrame(index=time_input,
                      columns=['move','vf','vfx','vfy','pn','p0','lon','lat','vmax'])

    st['move'] = gamma      # gamma, forward direction
    st['pn'] = 1013         # average pressure at the surface [mbar]
    st['p0'] = pmin         # minimum central pressure [mbar]
    st['vf'] = vmean        # translation speed [kt]    
    st['vfx'] = vmean * np.sin((gamma+180) * np.pi/180)   # [kt]
    st['vfy'] = vmean * np.cos((gamma+180) * np.pi/180)   # [kt]
    st['vmax'] = p1 * np.power(pmin,3) + p2 * np.power(pmin,2) + p3 * np.power(pmin,1) + p4   # [kt]
    st['lon'] = xt
    st['lat'] = yt

    # add some metadata
    st.x0 = x0
    st.y0 = y0
    st.R = R

    return st
