#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import radians, degrees, sin, cos, asin, acos, sqrt, atan2, pi

from .geo import gc_distance


# STOPMOTION LIBRARY

def Extract_basin_storms(xds, id_basin):
    'Selects storms with genesis in a given a basin (NA,SA,WP,EP,SP,NI,SI)'

    # select genesis basin (np.bytes_)
    origin_basin = xds.sel(date_time=xds.date_time[0]).basin.values
    origin_basin = np.asarray([c.decode('UTF-8') for c in origin_basin])

    # select storms with genesis basin
    storm_pos = np.where(origin_basin == id_basin)[0]

    # extract storms 
    xds_basin = xds.sel(storm=storm_pos)

    return xds_basin

def GeoAzimuth(lat1, lon1, lat2, lon2):
    'Returns geodesic azimuth between point1 and point2'

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    az = atan2(
        cos(lat2) * sin(lon2-lon1),
        cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2-lon1)
    )
    if lat1 <= -pi/2: az = 0
    if lat2 >=  pi/2: az = 0
    if lat2 <= -pi/2: az = pi
    if lat1 >=  pi/2: az = pi

    az = az % (2*pi)
    az = degrees(az)

    return az

# TODO check difference between GeoAzimuth
def calculate_azimut(lon_ini, lat_ini, lon_end, lat_end):
    '''
    '''

    if lon_ini < 0:    lon_ini += 360
    if lon_end < 0:    lon_end += 360

    gamma = GeoAzimuth(lat_ini, lon_ini, lat_end, lon_end)
    if gamma < 0.0: gamma += 360

    return gamma

def pres2wind(pres, xds_coef, center='WMO', basin='WP'):
    '''
    Returns empirical estimate for maximum wind speed (1-min average)
    '''

    # select coefficients
    coef = xds_coef.sel(center=center, basin=basin).coef.values

    # prediction
    u = coef
    X_test = pres
    y_pred = np.polyval(u, X_test)

    return y_pred    # vmax [kt]

def wind2rmw(vmax, lat):
    '''
    Returns radius of maximum winds RMW (Knaff et al. 2016)

    vmax    - maximum sustained winds [kt]
    lat     - latitude
    '''

    # constants
    pifac = np.arccos(-1) / 180     # pi/180

    # Vmax is used (at 10m surface and not the gradient wind!!)
    rm = 218.3784 - 1.2014*vmax + np.power(vmax/10.9844,2) - np.power(vmax/35.3052,3) - 145.509*np.cos(lat*pifac)  

    return rm   # nautical mile

def calculate_vmean(lon1, lat1, lon2, lat2, deltat):
    '''
    Returns mean translation speed between consecutive storm coordinates 
    deltat: timestep [hours]
    '''

    RE = 6378.135                                           # earth radius [km]
    arcl_h, gamma_h = gc_distance(lat2, lon2, lat1, lon1)   # great circle 

    r = arcl_h * np.pi / 180.0 * RE     # distance between consecutive coordinates [km]
    vmean = r / deltat                  # translation speed [km/h]

    vu = vmean * np.sin((gamma_h+180)*np.pi/180)
    vy = vmean * np.cos((gamma_h+180)*np.pi/180)

    return vmean, vu, vy    # [km/h]

def storm2stopmotion(df_storm, d_vns_center, xds_coef, 
                      var=['pressure'], varfill=['wind','rmw']): 
    '''
    df_storm    - storm dataframe
    d_vns_center- dictionary
    xds_coef    - dataset with polynomial fitting coefficients (Pmin-Vmax)

    var         - variables data (Pressure,Wind,Radii) 
    varfill     - variables to be filled if no available data (*Wind, **Radii)
                  * Wind is estimated using Pmin-Vmax empirical relationship for each basin
                  ** Radii is estimated using Knaff empiric estimate (function of Vmax and latitude)

    Generation of stopmotion 24h+6h segments from storm track:

    Warmup segment (24h total - 4 segments of 6h):
        4 segments, to define start/end coordinates, Vmean, relative angle
        4th segment (last) to define P, W, R, lat

    Target segment (1 segment of 6h):
        6h segment to define dP,dV,dW,dR,dAng

    Latitude is stored as (1) the original (mean) segment values and (2) the absolute value

    Relative angle between consecutive segments is calculated referenced to the geographic north,
    and those storms in the Southern hemisphere are multiplied by -1.
    '''

    # check for no conflict between var & varfill
    if np.intersect1d(var, varfill).size > 0:  
        print('WARNING: var and varfill cannot have duplicate variables')

    # dictionary parameters
    dict_lon = d_vns_center['longitude']
    dict_lat = d_vns_center['latitude']
    dict_pres = d_vns_center['pressure']
    dict_wind = d_vns_center['maxwinds']
    dict_rmw = d_vns_center['rmw']      # if None it will be calculated with Knaff 

    # storm variables, removing NaT
    time_st = df_storm.index.values
    pos = np.where(~np.isnat(time_st))[0]

    time = time_st[pos]
    lon = df_storm[dict_lon].values[pos]
    lat = df_storm[dict_lat].values[pos]
    pres = df_storm[dict_pres].values[pos]
    wind = df_storm[dict_wind].values[pos]
    if not d_vns_center['rmw']==None:  rmw = df_storm[dict_rmw].values[pos]
    if d_vns_center['rmw']==None:      rmw = pres*np.nan  # assign NaNs if radii is not provided

    # longitude convention
    lon[lon<0] += 360

    # store dataframe (storm coordinates)
    df0 = pd.DataFrame({
        'time': time,
        'longitude': lon,
        'latitude': lat,
        'pressure': pres,
        'wind': wind,
        'rmw': rmw,
    })

    # remove NaN for column variables in "var"
    df_nonan = df0.dropna(subset=var)

    # fill NaN values in wind
    # TODO storm basin needs to be defined
#     if 'wind' in varfill:       
#         # locate nan 'wind' and fill using Pmin-Vmax fitting coefficients (basins,centers)
#         pos_wind_nan = np.where(np.isnan(df_nonan.wind.values))
#         if pos_wind_nan[0].shape[0] > 0:
#             wind_nonan = pres2wind(df_nonan.pressure.values[pos_wind_nan], 
#                                    xds_coef, center=dict_center, basin=basin)
#             df_nonan['wind'].values[pos_wind_nan[0]] = wind_nonan

    # fill NaN values in rmw
    if 'rmw' in varfill:

        if d_vns_center['rmw']==None:    # no radii provided, fill with Knaff 
            df_nonan['rmw'] = wind2rmw(df_nonan.wind.values, 
                                        df_nonan.latitude.values)

        if not d_vns_center['rmw']==None:   # radii provided, fill NaNs with Knaff
            pos_rmw_nan = np.where(np.isnan(df_nonan.rmw.values))
            if pos_rmw_nan[0].shape[0] > 0: # if NaNs
                rmw_nonan = wind2rmw(df_nonan.wind.values[pos_rmw_nan], 
                                     df_nonan.latitude.values[pos_rmw_nan])
                df_nonan['rmw'].values[pos_rmw_nan[0]] = rmw_nonan

    # remove 'rmw' if not assigned (df_nonan must NOT have NaN)
    if not 'rmw' in var+varfill:    del df_nonan['rmw']

    # round time to hours
    df_nonan['time'] = pd.to_datetime(df_nonan['time'], format='%Y-%m-%d %H:%M:%S').dt.round('1h')

    # only keep 0,6,12,18 hours 
    hours = df_nonan['time'].dt.hour
    pos_hours = np.where((hours == 0) | (hours == 6) | (hours == 12) | (hours == 18))[0]
    df_dt = df_nonan.iloc[pos_hours]
    df_dt.index = df_dt['time']


    # generate stopmotion segments: consecutive 24h warmup + 6h target

    # (A) ---- WARMUP SEGMENT ----

    df_ = df_dt
    deltat = np.diff(df_.index) / np.timedelta64(1, 'h')   # timestep in hours

    # mean value of 24h (all 4 segments)  -->  Vmean, Azimuth
    vsegwarm = np.zeros((df_.index.shape)) * np.nan      # mean translational speed
    vxsegwarm = np.zeros((df_.index.shape)) * np.nan     # mean translational speed (dirx)
    vysegwarm = np.zeros((df_.index.shape)) * np.nan     # mean translational speed (diry)
    asegwarm = np.zeros((df_.index.shape)) * np.nan      # azimuth respect geographic North

    # mean value of last 6h segment  -->  Pmean,Wmean,Rmean,Latmean
    pseg = np.zeros((df_.index.shape)) * np.nan      # mean Pmin of consecutive nodes
    wseg = np.zeros((df_.index.shape)) * np.nan      # mean maximum wind of consecutive nodes
    rseg = np.zeros((df_.index.shape)) * np.nan      # mean radius of maximum wind of consecutive nodes
    lseg = np.zeros((df_.index.shape)) * np.nan      # mean latitude of first/origin segment node
    laseg = np.zeros((df_.index.shape)) * np.nan     # absolute latitude of first/origin segment node
    vseg = np.zeros((df_.index.shape)) * np.nan      # mean translational speed
    vxseg = np.zeros((df_.index.shape)) * np.nan     # mean translational speed (dirx)
    vyseg = np.zeros((df_.index.shape)) * np.nan     # mean translational speed (diry)
    aseg = np.zeros((df_.index.shape)) * np.nan      # azimuth respect geographic North

    for i in np.arange(3, deltat.size-1):  # starting at the first 4th segment

        # TODO generalize creation of HYBRID warmup segments if not consecutive segments (gaps)

        if (deltat[i]==6) & (deltat[i-3:i].sum()==18):   # if 4 consecutive segments

            # Vmean (mean of all 4 segments)
            vsegwarm[i+1], vxsegwarm[i+1], vysegwarm[i+1] = calculate_vmean(
                df_['longitude'].values[i-3],
                df_['latitude'].values[i-3],
                df_['longitude'].values[i+1],
                df_['latitude'].values[i+1],
                deltat[i-3:i+1].sum(),
            )

            # Azimuth (mean of all 4 segments)
            asegwarm[i+1] = calculate_azimut(
                df_['longitude'].values[i-3], df_['latitude'].values[i-3],
                df_['longitude'].values[i+1], df_['latitude'].values[i+1],
            )  # lon_ini, lat_ini, lon_end, lat_end

            # Pmean,Wmean,Rmean,LatMean (last 4th segment)
            pseg[i+1] = np.mean([df_['pressure'].values[i], df_['pressure'].values[i+1]])
            wseg[i+1] = np.mean([df_['wind'].values[i], df_['wind'].values[i+1]])
            lseg[i+1] = np.mean([df_['latitude'].values[i], df_['latitude'].values[i+1]])
            laseg[i+1] = np.abs(lseg[i+1])
            if 'rmw' in df_.keys():    rseg[i+1] = np.mean([df_['rmw'].values[i], df_['rmw'].values[i+1]])

            # Vmean, Azimuth (last 4th segment)
            vseg[i+1], vxseg[i+1], vyseg[i+1] = calculate_vmean(
                df_['longitude'].values[i], df_['latitude'].values[i],
                df_['longitude'].values[i+1], df_['latitude'].values[i+1],
                deltat[i],
            )

            aseg[i+1] = calculate_azimut(
                df_['longitude'].values[i], df_['latitude'].values[i],
                df_['longitude'].values[i+1], df_['latitude'].values[i+1],
            ) # lon_ini, lat_ini, lon_end, lat_end

    # add to dataframe (warmup segment)
    df_['pseg'] = pseg              # Pmean last segment
    df_['wseg'] = wseg              # Wmean last segment [kt]
    df_['lseg'] = lseg              # latitude last segment [º]
    df_['laseg'] = laseg            # absolute latitude last segment [º]
    if 'rmw' in df_.keys():   df_['rseg'] = rseg    # radii last segment [nmile]
    df_['vsegwarm'] = vsegwarm      # Vmean all segments [km/h]
    df_['vxsegwarm'] = vxsegwarm
    df_['vysegwarm'] = vysegwarm
    df_['asegwarm'] = asegwarm      # Azimuth all segments [º]
    df_['vseg'] = vseg              # Vmean last segment
    df_['vxseg'] = vxseg
    df_['vyseg'] = vyseg
    df_['aseg'] = aseg              # Azimuth last segment


    # (B) ---- TARGET SEGMENT ----

    # calculate change rates between warmup and target segments
    dpseg = np.zeros((df_.index.shape)) * np.nan      # Pmin variation
    dlseg = np.zeros((df_.index.shape)) * np.nan      # latitude variation
    dlaseg = np.zeros((df_.index.shape)) * np.nan     # absolute latitude variation
    dwseg = np.zeros((df_.index.shape)) * np.nan      # wind variation
    if 'rmw' in df_.keys():   drseg = np.zeros((df_.index.shape)) * np.nan      # rmw variation
    dvseg = np.zeros((df_.index.shape)) * np.nan      # mean translational speed variation
    dvxseg = np.zeros((df_.index.shape)) * np.nan     # mean translational speed variation
    dvyseg = np.zeros((df_.index.shape)) * np.nan     # mean translational speed variation
    daseg = np.zeros((df_.index.shape)) * np.nan      # azimuth variation

    for i in range(pseg.size - 1):

        # consecutive segments (target - last warmup)
        dpseg[i] = df_.pseg.values[i+1] - df_.pseg.values[i]     # [mbar]
        dlseg[i] = df_.lseg.values[i+1] - df_.lseg.values[i]     # [º]
        dlaseg[i] = df_.laseg.values[i+1] - df_.laseg.values[i]  # [º]
        dwseg[i] = df_.wseg.values[i+1] - df_.wseg.values[i]     # [kt]
        if 'rmw' in df_.keys():   drseg[i] = df_.rseg.values[i+1] - df_.rseg.values[i]   # [nmile]

        # "non-consecutive" segments (target - whole warmup)
        dvseg[i] = df_.vseg.values[i+1] - df_.vsegwarm.values[i]     # [km/h]
        dvxseg[i] = df_.vxseg.values[i+1] - df_.vxsegwarm.values[i]  # [km/h]
        dvyseg[i] = df_.vyseg.values[i+1] - df_.vysegwarm.values[i]  # [km/h]

        # angle variation
        ang1, ang2 = df_.asegwarm.values[i], df_.aseg.values[i+1]
        delta_ang = ang2 - ang1             # [º]
        sign = np.sign(df_.lseg.values[i])  # hemisphere factor: north (+), south (-)

        if (ang2 > ang1) & (delta_ang < 180):      daseg[i] = sign * (delta_ang)
        elif (ang2 > ang1) & (delta_ang > 180):    daseg[i] = sign * (delta_ang - 360)
        elif (ang2 < ang1) & (delta_ang > -180):   daseg[i] = sign * (delta_ang)
        elif (ang2 < ang1) & (delta_ang < -180):   daseg[i] = sign * (delta_ang + 360)

    # add to dataframe (target segment)
    df_['dpseg'] = dpseg            # Pmin variation
    df_['dwseg'] = dwseg            # Wind variation
    df_['dlseg'] = dlseg            # Latitude variation
    df_['dlaseg'] = dlaseg          # Absolute latitude variation
    if 'rmw' in df_.keys():   df_['drseg'] = drseg  # Radii variation
    df_['dvseg'] = dvseg            # Vmean variation
    df_['dvxseg'] = dvxseg
    df_['dvyseg'] = dvyseg
    df_['daseg'] = daseg            # Azimuth variation

    return df_

def stopmotion_interpolation(df_seg, st, t_warm=24, t_seg=6, t_prop=42):
    '''
    SWAN is executed in cartesian coordinates for SHyTCWaves

    Dataframe provides the following parameters:
    - 24h warmup segments
        vsegwarm:  mean translational velocity (km/h)
        asegwarm:  azimut (º)
    - 6h last warmup segment
        pseg:      minimum central pressure (mbar)
        lseg:      latitude (º)
        laseg:     absolute latitude (º)
        wseg:      maximum winds (kt)
        rseg:      radii at maximum winds (nmile)
    - 6h target segment
        dpseg:     variation of minimum central pressure
        dvseg:     variation of translational velocity
        daseg:     variation of azimut
        dlseg:     variation of latitude
        dlaseg:    variation of absolute latitude
        dwseg:     variation of maximum winds
        drseg:     variation of radii at maximum winds

    Returns for each row (event), the storm coordinates for a 3-days period:
        24h for the first warm-up segment ending at (x,y)=(0,0)
        6h for the second segment starting at (x,y)=(0,0)
        42h with no track coordinates (no wind forcing)
    '''

    # remove NaNs
    df = df_seg.dropna()

    N = df['dpseg'].size                # number of events
    sign = np.sign(st['lat'][0])        # storm latitude sign (hemisphere)

    # TODO set computational time according to Vmean criteria
    dt_comp = 20                        # computational timestep [minutes] 

    # time array for SWAN input
    ts = t_warm + t_seg + t_prop        # total duration [h]
    ts = np.asarray(ts) * 60 / dt_comp  # number of intervals for computation

    # random initial date (target segment is set to begin at 2020-01-01)
    date_ini = pd.Timestamp(2019, 12, 31, 0)
    time_input0 = pd.date_range(date_ini, periods=int(ts), 
                                freq='{0}MIN'.format(dt_comp))
    time_input = np.array(time_input0)

    # list of dataframes for SWAN cases
    st_list, we_list = [], []

    for i in range(N):
        # vortex input variables: (time,x,y,vmean,ut,vt,pn,p0,vmax,R)
        x, y, vmean, ut, vt, p0, vmax, rmw, lat = [],[],[],[],[],[],[],[],[]

        seg_i = df.iloc[i]

        # (A) ---- WARMUP SEGMENT ----
        # 24h period before reaching (x,y)=(0,0), coordinates over the x-axis (y=0)

        # initial coordinate according to vmean
        x0 = - seg_i['vsegwarm']*24*10**3    # distance (m) in 24h

        # interpolation of coordinates at every timestep (dt_comp)
        interval_warm = int(t_warm * 60 / dt_comp)
        for j in range(interval_warm):
            if j==0:    x.append(x0)
            else:       x.append(x[-1] + seg_i['vsegwarm']*(dt_comp/60)*10**3)
            y.append(0)                        # m
            vmean.append(seg_i['vsegwarm'])    # km/h
            ut.append(seg_i['vsegwarm'])       # km/h
            vt.append(0)                       # km/h
            p0.append(seg_i['pseg'])           # mbar
            vmax.append(seg_i['wseg'])         # kt
            rmw.append(seg_i['rseg'])          # nmile
            lat.append(seg_i['laseg'] + seg_i['dlaseg'])  # º (target segment)

        # (B) ---- TARGET SEGMENT ----
        # 6h period starting at (x,y)=(0,0) acording to azimuth variation

        # calculate vmean components
        vel = seg_i['vsegwarm'] + seg_i['dvseg']     # km/h
        velx = vel * np.sin((seg_i['daseg']*sign + 90)*np.pi/180)
        vely = vel * np.cos((seg_i['daseg']*sign + 90)*np.pi/180)

        interval_seg = int(t_seg * 60 / dt_comp)
        for k in range(interval_seg):
            x.append(x[-1] + velx*(dt_comp/60)*10**3)
            y.append(y[-1] + vely*(dt_comp/60)*10**3)
            vmean.append(vel)
            ut.append(velx)
            vt.append(vely)
            p0.append(seg_i['pseg'] + seg_i['dpseg'])
            vmax.append(seg_i['wseg'] + seg_i['dwseg'])
            rmw.append(seg_i['rseg'] + seg_i['drseg'])
            lat.append(seg_i['laseg'] + seg_i['dlaseg'])

        # (C) ---- NO FORCING ---- 
        # propagation period: fill with NaNs
        interval_prop = int(t_prop * 60 / dt_comp)
        for z in range(interval_prop):
            x.append(np.nan)
            y.append(np.nan)
            vmean.append(np.nan)
            ut.append(np.nan)
            vt.append(np.nan)
            p0.append(np.nan)
            vmax.append(np.nan)
            rmw.append(np.nan)
            lat.append(np.nan)

        # store dataframe
        st_seg = pd.DataFrame(
            index=time_input,
            columns=['x','y','vf','vfx','vfy','pn','p0','vmax','rmw','lat'],
        )

        st_seg['x'] = np.array(x)               # m
        st_seg['y'] = np.array(y)               # m
        st_seg['lon'] = np.array(x)             # m (idem for plots)
        st_seg['lat'] = np.array(y)             # m (idem for plots)
        st_seg['vf'] = np.array(vmean)/1.852    # km/h to kt
        st_seg['vfx'] = np.array(ut)/1.852      # km/h to kt
        st_seg['vfy'] = np.array(vt)/1.852      # km/h to kt
        st_seg['pn'] = 1013                     # mbar
        st_seg['p0'] = np.array(p0)             # mbar
        st_seg['vmax'] = np.array(vmax)         # kt
        st_seg['rmw'] = np.array(rmw)           # nmile
        st_seg['latitude'] = np.array(lat)*sign # º

        st_list.append(st_seg)

        # generate wave event (empty)
        we = pd.DataFrame(index=time_input, columns=['hs', 't02', 'dir', 'spr', 'U10', 'V10'])
        we['level'] = 0
        we['tide'] = 0
        we_list.append(we)

    return st_list, we_list

def segments_database_center(xds_ibtracsv4, d_vns_center, xds_coef, 
                             var=['pressure'], varfill=['wind','rmw']): 
    '''
    xds         - historical storm records
    d_vns_center- dictionary
    xds_coef    - dataset with polynomial fitting coefficients (Pmin-Vmax)

    var         - variables data (Pressure,Wind,Radii) 
    varfill     - variables to be filled if no available data (*Wind, **Radii)
                  * Wind is estimated using Pmin-Vmax empirical relationship for each basin
                  ** Radii is estimated using Knaff empiric estimate (function of Vmax and latitude)

    Latitude is stored as (1) the original (mean) segment values and (2) the absolute value

    Relative angle between consecutive segments is calculated referenced to the geographic north,
    and those storms in the Southern hemisphere are multiplied by -1.
    '''

    # check for no conflict between var & varfill
    if np.intersect1d(var, varfill).size > 0:
        print('WARNING: var and varfill cannot have duplicate variables')

#    import time
#    start_time = time.time()

    # set storms id as coordinate
    xds_ibtracsv4['stormid'] = (('storm'), xds_ibtracsv4.storm.values)
    xds_ibtracsv4.set_coords('stormid')

    # dictionary parameters
    dict_center = d_vns_center['source']
    dict_basin = d_vns_center['basins']
    dict_lon = d_vns_center['longitude']
    dict_lat = d_vns_center['latitude']
    dict_pres = d_vns_center['pressure']
    dict_wind = d_vns_center['wind']
    dict_rmw = d_vns_center['rmw']      # if None it will be calculated with Knaff 

    # loop for all basins
    df_center_ls = []
    num_st_basin, num_st_mask, num_st_nonan, num_st_dt = [],[],[],[]
    num_st_ids, num_st_segs, num_st_dseg, num_st = [],[],[],[]

    for basin in dict_basin:

        # extract storms at basin X
        xds_basin = Extract_basin_storms(xds_ibtracsv4, basin)

        # extract storm coordinate values (center X: WMO, USA, BOM)
        time = np.array([], dtype=np.datetime64)
        basinid = np.array([],dtype=np.int64)
        stid, centerid, dist2land = np.array([]), np.array([]), np.array([])
        lon, lat, pres, wind, rmw = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        for i in range(xds_basin.storm.size):
            time_i = xds_basin.time.values[i,:]
            pos = np.where(~np.isnat(time_i))[0]

            time = np.concatenate((time, time_i[pos]))
            stid = np.concatenate((stid, np.asarray(pos.size * [xds_basin.stormid.values[i]])))
            centerid = np.concatenate((centerid, np.asarray(pos.size * [dict_center])))
            basinid = np.concatenate((basinid, np.asarray(pos.size * [basin])))
            dist2land = np.concatenate((dist2land, xds_basin.dist2land.values[i,:][pos]))
            lon = np.concatenate((lon, xds_basin[dict_lon].values[i,:][pos]))
            lat = np.concatenate((lat, xds_basin[dict_lat].values[i,:][pos]))

            # variables (pressure is always taken)
            pres = np.concatenate((pres, xds_basin[dict_pres].values[i,:][pos]))
            wind = np.concatenate((wind, xds_basin[dict_wind].values[i,:][pos]))
            if not d_vns_center['rmw']==None:
                rmw = np.concatenate((rmw, xds_basin[dict_rmw].values[i,:][pos]))

        # fill NaNs when radii is not provided
        if d_vns_center['rmw']==None:       rmw = pres*np.nan

        # assign NaNs to dist2land=0 (landmask)
        dist2land[np.where(dist2land==0)[0]] = np.nan

        # longitude convention
        lon[lon<0] += 360

        # store dataframe
        df0 = pd.DataFrame({
            'time': time,
            'center': centerid,
            'basin': basinid, 
            'id': stid,
            'dist2land': dist2land,
            'longitude': lon,
            'latitude': lat,
            'pressure': pres,
            'wind': wind,
            'rmw': rmw,
        })

        # remove NaN on specific column
        df_mask = df0.dropna(subset=['dist2land'])
        df_nonan = df_mask.dropna(subset=var)    # variables in "var" are kept

        # fill NaN values with Pmin-Vmax coefficients (basins, centers)
        if 'wind' in varfill:
            pos_wind_nan = np.where(np.isnan(df_nonan.wind.values))
            if pos_wind_nan[0].shape[0] > 0:
                wind_nonan = pres2wind(df_nonan.pressure.values[pos_wind_nan],
                                       xds_coef, center=dict_center, basin=basin)
                df_nonan['wind'].values[pos_wind_nan[0]] = wind_nonan

        # fill values with Knaff estimates
        if 'rmw' in varfill:
            if d_vns_center['rmw']==None:
                df_nonan['rmw'] = wind2rmw(
                    df_nonan.wind.values,
                    df_nonan.latitude.values)

            # fill NaN even when those centers have rmw data
            if not d_vns_center['rmw']==None:
                pos_rmw_nan = np.where(np.isnan(df_nonan.rmw.values))
                if pos_rmw_nan[0].shape[0] > 0:
                    rmw_nonan = wind2rmw(df_nonan.wind.values[pos_rmw_nan],
                                         df_nonan.latitude.values[pos_rmw_nan])
                    df_nonan['rmw'].values[pos_rmw_nan[0]] = rmw_nonan

        # remove variables not included (df_nonan must have NO nans)
        if not 'rmw' in var+varfill:    del df_nonan['rmw']

        # round time to hours
        df_nonan['time'] = pd.to_datetime(df_nonan['time'], format='%Y-%m-%d %H:%M:%S').dt.round('1h')

        # only keep hours 0,6,12,18
        hours = df_nonan['time'].dt.hour
        pos_hours = np.where((hours == 0) | (hours == 6) | (hours == 12) | (hours == 18))[0]
        df_dt = df_nonan.iloc[pos_hours]
        df_dt.index = df_dt['time']

        # store number nodes
        num_st_basin.append(df0.shape[0])
        num_st_mask.append(df_mask.shape[0])
        num_st_nonan.append(df_nonan.shape[0])
        num_st_ids.append(np.unique(df_dt.id).size)
        num_st_dt.append(df_dt.shape[0])

        # generate segments database (segment = 2 consecutive nodes)
        df_ = df_dt
        deltat = np.diff(df_.index) / np.timedelta64(1, 'h')   # timestep in hours

        pseg = np.zeros((df_.index.shape)) * np.nan      # mean Pmin of consecutive nodes
        vseg = np.zeros((df_.index.shape)) * np.nan      # mean translational speed
        vxseg = np.zeros((df_.index.shape)) * np.nan     # mean translational speed (dirx)
        vyseg = np.zeros((df_.index.shape)) * np.nan     # mean translational speed (diry)
        aseg = np.zeros((df_.index.shape)) * np.nan      # azimuth or segment angle respect North (gamma)
        lseg = np.zeros((df_.index.shape)) * np.nan      # latitude of first/origin segment node
        laseg = np.zeros((df_.index.shape)) * np.nan     # absolute latitude of first/origin segment node
        rseg = np.zeros((df_.index.shape)) * np.nan      # mean radius of maximum wind of consecutive nodes
        wseg = np.zeros((df_.index.shape)) * np.nan      # mean maximum wind of consecutive nodes

        for i in range(deltat.size):
            storm_i1 = df_.id.values[i]
            storm_i2 = df_.id.values[i+1]

            if (deltat[i]==6) & (storm_i1==storm_i2):
                pseg[i] = np.mean([df_['pressure'].values[i], df_['pressure'].values[i+1]])
                vseg[i], vxseg[i], vyseg[i] = calculate_vmean(
                    df_['longitude'].values[i], df_['latitude'].values[i],
                    df_['longitude'].values[i+1], df_['latitude'].values[i+1],
                    deltat[i],
                )

                aseg[i] = calculate_azimut(
                    df_['longitude'].values[i], df_['latitude'].values[i],
                    df_['longitude'].values[i+1], df_['latitude'].values[i+1],
                ) # lon_ini, lat_ini, lon_end, lat_end

                lseg[i] = df_['latitude'].values[i]
                laseg[i] = np.abs(df_['latitude'].values[i])
                wseg[i] = np.mean([df_['wind'].values[i], df_['wind'].values[i+1]])
                if 'rmw' in df_.keys():    rseg[i] = np.mean([df_['rmw'].values[i], df_['rmw'].values[i+1]])

        # add to dataframe
        df_['pseg'] = pseg          # mbar
        df_['vseg'] = vseg          # km/h
        df_['vxseg'] = vxseg        # km/h
        df_['vyseg'] = vyseg        # km/h
        df_['aseg'] = aseg          # º
        df_['lseg'] = lseg          # º
        df_['laseg'] = laseg        # º
        df_['wseg'] = wseg          # kt
        if 'rmw' in df_.keys():   df_['rseg'] = rseg    # nmile

        # store number nodes
        num_st_segs.append(df_.dropna().shape[0])

        # calculate consecutive segments variations
        dpseg = np.zeros((df_.index.shape)) * np.nan      # mean Pmin variation
        dvseg = np.zeros((df_.index.shape)) * np.nan      # mean translational variation
        dvxseg = np.zeros((df_.index.shape)) * np.nan     # mean translational variation
        dvyseg = np.zeros((df_.index.shape)) * np.nan     # mean translational variation
        daseg = np.zeros((df_.index.shape)) * np.nan      # azimuth variation
        dlseg = np.zeros((df_.index.shape)) * np.nan      # latitude variation
        dlaseg = np.zeros((df_.index.shape)) * np.nan     # absolute latitude variation
        dwseg = np.zeros((df_.index.shape)) * np.nan      # wind variation
        if 'rmw' in df_.keys():   drseg = np.zeros((df_.index.shape)) * np.nan      # rmw variation

        for i in range(pseg.size - 1):
            storm_i1 = df_.id.values[i]
            storm_i2 = df_.id.values[i+1]

            if storm_i1==storm_i2:
                dpseg[i] = df_.pseg.values[i+1] - df_.pseg.values[i]     # mbar 
                dvseg[i] = df_.vseg.values[i+1] - df_.vseg.values[i]     # km/h 
                dvxseg[i] = df_.vxseg.values[i+1] - df_.vxseg.values[i]  # km/h 
                dvyseg[i] = df_.vyseg.values[i+1] - df_.vyseg.values[i]  # km/h 
                dlseg[i] = df_.lseg.values[i+1] - df_.lseg.values[i]     # º 
                dlaseg[i] = df_.laseg.values[i+1] - df_.laseg.values[i]  # º 
                dwseg[i] = df_.wseg.values[i+1] - df_.wseg.values[i]     # kt
                if 'rmw' in df_.keys():   drseg[i] = df_.rseg.values[i+1] - df_.rseg.values[i]   # nmile

                # hemisphere sign factor for angle variations: north (+), south (-)
                sign = np.sign(df_.lseg.values[i])

                # angle variation
                ang1 = df_.aseg.values[i]
                ang2 = df_.aseg.values[i+1]
                delta_ang = ang2 - ang1  # º
                if (ang2 > ang1) & (delta_ang < 180):      daseg[i] = sign * (delta_ang)
                elif (ang2 > ang1) & (delta_ang > 180):    daseg[i] = sign * (delta_ang - 360)
                elif (ang2 < ang1) & (delta_ang > -180):   daseg[i] = sign * (delta_ang)
                elif (ang2 < ang1) & (delta_ang < -180):   daseg[i] = sign * (delta_ang + 360)

        # add to dataframe
        df_['dpseg'] = dpseg
        df_['dvseg'] = dvseg
        df_['dvxseg'] = dvxseg
        df_['dvyseg'] = dvyseg
        df_['daseg'] = daseg
        df_['dlseg'] = dlseg
        df_['dlaseg'] = dlaseg
        df_['dwseg'] = dwseg
        if 'rmw' in df_.keys():   df_['drseg'] = drseg

        # store number nodes
        num_st_dseg.append(df_.dropna().shape[0])
        num_st.append(np.unique(df_.dropna().id).size)

        # store (order 'NA','SA','WP','EP','SP','NI','SI')
        df_center_ls.append(df_)

    # store nums
    df_num = pd.DataFrame(
        {
        'n_basin': np.array(num_st_basin),
        'n_mask': np.array(num_st_mask),
        'n_nonan': np.array(num_st_nonan),
        'n_ids': np.array(num_st_ids),
        'n_6h': np.array(num_st_dt),
        'n_seg': np.array(num_st_segs),
        'n_dseg': np.array(num_st_dseg),
        'n_storms': np.array(num_st),
        },
        index = dict_basin,
    )

    return df_center_ls, df_num

#def storm2segment(xds_storm, d_vns_center, xds_coef, 
#                  var=['pressure'], varfill=['wind','rmw']): 
#    '''
#    Generation of database following different criteria:
#    var        - variables data (Pressure,Wind,Radii) 
#    varfill    - variables to be filled if no data (Wind,Radii)
#    
#    * Wind is estimated using Pmin-Vmax empirical relationship for each basin
#    * Radii is estimated using Knaff empiric estimate (function of Vmax and latitude)
#    
#    Latitude is stored both the original (mean) segment values and the absolute value
#    Relative angle between consecutive segments is calculated referenced to the geographic north,
#    and those storms in the Southern hemisphere are multiplied by -1.
#    '''
#    
#    if np.intersect1d(var, varfill).size > 0:  
#        print('WARNING: var and varfill cannot have duplicate variables')
#    
#    # set storms_id as coordinate
#    xds_storm['stormid'] = (('storm'), xds_storm.storm.values)
#    xds_storm.set_coords('stormid')
#
#    # dictionary parameters
#    dict_center = d_vns_center['source']
#    dict_basin = d_vns_center['basins']
#    dict_lon = d_vns_center['longitude']
#    dict_lat = d_vns_center['latitude']
#    dict_pres = d_vns_center['pressure']
#    dict_wind = d_vns_center['wind']
#    dict_rmw = d_vns_center['rmw']      # if None it will be calculated with Knaff function (2015)
#    
#    # storm variables
#    time_st = xds_storm.time.values
#    pos = np.where(~np.isnat(time_st))[0]
#    
#    time = time_st[pos]
#    basinid = xds_storm.basin.values[pos]
#    dist2land = xds_storm.dist2land.values[pos]
#    lon = xds_storm[dict_lon].values[pos]
#    lat = xds_storm[dict_lat].values[pos]
#    
#    pres = xds_storm[dict_pres].values[pos]
#    wind = xds_storm[dict_wind].values[pos]
#    if not d_vns_center['rmw']==None:  rmw = xds_storm[dict_rmw].values[pos]
#    if d_vns_center['rmw']==None:      rmw = pres*np.nan  # NaNs when radii is not provided
#
#
##    # loop for all basins
##    df_center_ls = []
##    num_st_basin, num_st_mask, num_st_nonan, num_st_dt = [],[],[],[]
##    num_st_ids, num_st_segs, num_st_dseg, num_st = [],[],[],[]
#
#    # extract storms at basin X
##    for basin in dict_basin:
#        
##        xds_basin = Extract_basin_storms(xds_ibtracsv4, basin)
#
##        # extract storm coordinates values (center X: WMO, USA, BOM)
##        time = np.array([], dtype=np.datetime64)
##        basinid = np.array([],dtype=np.int64)
##        stid, centerid, dist2land = np.array([]), np.array([]), np.array([])
##        lon, lat, pres, wind, rmw = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
##
##        for i in range(xds_basin.storm.size):
##            time_i = xds_basin.time.values[i,:]
##            pos = np.where(~np.isnat(time_i))[0]
##            
##            time = np.concatenate((time, time_i[pos]))
##            stid = np.concatenate((stid, np.asarray(pos.size * [xds_basin.stormid.values[i]])))
##            centerid = np.concatenate((centerid, np.asarray(pos.size * [dict_center])))
##            basinid = np.concatenate((basinid, np.asarray(pos.size * [basin])))
##            dist2land = np.concatenate((dist2land, xds_basin.dist2land.values[i,:][pos]))
##            lon = np.concatenate((lon, xds_basin[dict_lon].values[i,:][pos]))
##            lat = np.concatenate((lat, xds_basin[dict_lat].values[i,:][pos]))
##            
##            # variables (pressure is always taken)
##            pres = np.concatenate((pres, xds_basin[dict_pres].values[i,:][pos]))
##            wind = np.concatenate((wind, xds_basin[dict_wind].values[i,:][pos]))
##            if not d_vns_center['rmw']==None:  
##                rmw = np.concatenate((rmw, xds_basin[dict_rmw].values[i,:][pos]))
##                
##        # fill NaNs when radii is not provided
##        if d_vns_center['rmw']==None:          
##            rmw = pres*np.nan
#
#        # assign NaNs to dist2land=0 (landmask)
#        dist2land[np.where(dist2land==0)[0]] = np.nan
#
#        # longitude convention
#        lon[lon<0] += 360
#
#        # store dataframe
#        df0 = pd.DataFrame({
#            'time': time,
#            'center': centerid,
#            'basin': basinid, 
#            'id': stid,
#            'dist2land': dist2land,
#            'longitude': lon,
#            'latitude': lat,
#            'pressure': pres,
#            'wind': wind,
#            'rmw': rmw,
#        })
#
#        # remove NaN on specific column
#        df_mask = df0.dropna(subset=['dist2land'])
#        df_nonan = df_mask.dropna(subset=var)    # variables in "var" are kept
#        
#        # fill nan values 
#        if 'wind' in varfill:       
#            # locate nan 'wind' and fill using Pmin-Vmax fitting coefficients (basins,centers)
#            pos_wind_nan = np.where(np.isnan(df_nonan.wind.values))
#            if pos_wind_nan[0].shape[0] > 0:
#                wind_nonan = pres2wind(df_nonan.pressure.values[pos_wind_nan], 
#                                       xds_coef, center=dict_center, basin=basin)
#                df_nonan['wind'].values[pos_wind_nan[0]] = wind_nonan
#
#        if 'rmw' in varfill:    
#            # locate nan 'rmw' and fill using Knaff estimate for those centers without radii
#            if d_vns_center['rmw']==None:          
#                df_nonan['rmw'] = wind2rmw(df_nonan.wind.values, 
#                                            df_nonan.latitude.values)
#                
#            # check if there are NaN radius even when those centers have rmw data
#            if not d_vns_center['rmw']==None:      
#                pos_rmw_nan = np.where(np.isnan(df_nonan.rmw.values))
#                if pos_rmw_nan[0].shape[0] > 0:
#                    rmw_nonan = wind2rmw(df_nonan.wind.values[pos_rmw_nan], 
#                                         df_nonan.latitude.values[pos_rmw_nan])
#                    df_nonan['rmw'].values[pos_rmw_nan[0]] = rmw_nonan
#        
#        # remove variables not included (df_nonan must have NO nans)
#        if not 'rmw' in var+varfill:    del df_nonan['rmw']
#            
#        # round time to hours
#        df_nonan['time'] = pd.to_datetime(df_nonan['time'], 
#                                          format='%Y-%m-%d %H:%M:%S').dt.round('1h')
#        
#        # only keep hours 0,6,12,18
#        hours = df_nonan['time'].dt.hour
#        pos_hours = np.where((hours == 0) | (hours == 6) | (hours == 12) | (hours == 18))[0]
#        df_dt = df_nonan.iloc[pos_hours]
#        df_dt.index = df_dt['time']
#
#        # store number nodes
#        num_st_basin.append(df0.shape[0])
#        num_st_mask.append(df_mask.shape[0])
#        num_st_nonan.append(df_nonan.shape[0])
#        num_st_ids.append(np.unique(df_dt.id).size)
#        num_st_dt.append(df_dt.shape[0])
#
#        # generate segments database (segment = 2 consecutive nodes)
#        df_ = df_dt
#
#        pseg = np.zeros((df_.index.shape)) * np.nan      # mean Pmin of consecutive nodes
#        vseg = np.zeros((df_.index.shape)) * np.nan      # mean translational speed
#        vxseg = np.zeros((df_.index.shape)) * np.nan     # mean translational speed (dirx)
#        vyseg = np.zeros((df_.index.shape)) * np.nan     # mean translational speed (diry)
#        aseg = np.zeros((df_.index.shape)) * np.nan      # azimut or segment angle respect North (gamma)
#        lseg = np.zeros((df_.index.shape)) * np.nan      # latitude of first/origin segment node
#        laseg = np.zeros((df_.index.shape)) * np.nan     # absolute latitude of first/origin segment node
#        rseg = np.zeros((df_.index.shape)) * np.nan      # mean radius of maximum wind of consecutive nodes
#        wseg = np.zeros((df_.index.shape)) * np.nan      # mean maximum wind of consecutive nodes
#        deltat = np.diff(df_.index) / np.timedelta64(1, 'h')   # timestep in hours
#
#        for i in range(deltat.size):
#            storm_i1 = df_.id.values[i]
#            storm_i2 = df_.id.values[i+1]
#
#            if (deltat[i]==6) & (storm_i1==storm_i2):
#                pseg[i] = np.mean([df_['pressure'].values[i], df_['pressure'].values[i+1]])       
#                vseg[i], vxseg[i], vyseg[i] = calculate_vmean(df_['longitude'].values[i], df_['latitude'].values[i],   
#                                                              df_['longitude'].values[i+1], df_['latitude'].values[i+1], 
#                                                              deltat[i])
#                aseg[i] = calculate_azimut(df_['longitude'].values[i], df_['latitude'].values[i],  
#                                           df_['longitude'].values[i+1], df_['latitude'].values[i+1])  
#                                            # lon_ini, lat_ini, lon_end, lat_end
#                lseg[i] = df_['latitude'].values[i]  
#                laseg[i] = np.abs(df_['latitude'].values[i])  
#                wseg[i] = np.mean([df_['wind'].values[i], df_['wind'].values[i+1]])
#                if 'rmw' in df_.keys():    rseg[i] = np.mean([df_['rmw'].values[i], df_['rmw'].values[i+1]])
#
#        # add to dataframe
#        df_['pseg'] = pseg
#        df_['vseg'] = vseg
#        df_['vxseg'] = vxseg
#        df_['vyseg'] = vyseg
#        df_['aseg'] = aseg
#        df_['lseg'] = lseg
#        df_['laseg'] = laseg
#        df_['wseg'] = wseg
#        if 'rmw' in df_.keys():   df_['rseg'] = rseg
#        
#
#        # store number nodes
#        num_st_segs.append(df_.dropna().shape[0])
#
#        # calculate consecutive segments variations
#        dpseg = np.zeros((df_.index.shape)) * np.nan      # mean Pmin variation
#        dvseg = np.zeros((df_.index.shape)) * np.nan      # mean translational variation
#        dvxseg = np.zeros((df_.index.shape)) * np.nan     # mean translational variation
#        dvyseg = np.zeros((df_.index.shape)) * np.nan     # mean translational variation
#        daseg = np.zeros((df_.index.shape)) * np.nan      # azimut variation
#        dlseg = np.zeros((df_.index.shape)) * np.nan      # latitude variation
#        dlaseg = np.zeros((df_.index.shape)) * np.nan     # absolute latitude variation
#        dwseg = np.zeros((df_.index.shape)) * np.nan      # wind variation
#        if 'rmw' in df_.keys():   drseg = np.zeros((df_.index.shape)) * np.nan      # rmw variation
#        
#        for i in range(pseg.size - 1):
#            storm_i1 = df_.id.values[i]
#            storm_i2 = df_.id.values[i+1]
#
#            if storm_i1==storm_i2:
#                dpseg[i] = df_.pseg.values[i+1] - df_.pseg.values[i]     # mbar 
#                dvseg[i] = df_.vseg.values[i+1] - df_.vseg.values[i]     # km/h 
#                dvxseg[i] = df_.vxseg.values[i+1] - df_.vxseg.values[i]  # km/h 
#                dvyseg[i] = df_.vyseg.values[i+1] - df_.vyseg.values[i]  # km/h 
#                dlseg[i] = df_.lseg.values[i+1] - df_.lseg.values[i]     # º 
#                dlaseg[i] = df_.laseg.values[i+1] - df_.laseg.values[i]  # º 
#                dwseg[i] = df_.wseg.values[i+1] - df_.wseg.values[i]     # kt
#                if 'rmw' in df_.keys():   drseg[i] = df_.rseg.values[i+1] - df_.rseg.values[i]   # nmile
#                
#                # hemisphere sign factor for angle variations: north (+), south (-)
#                sign = np.sign(df_.lseg.values[i])
#
#                # angle variation
#                ang1 = df_.aseg.values[i]
#                ang2 = df_.aseg.values[i+1]
#                delta_ang = ang2 - ang1  # º
#                if (ang2 > ang1) & (delta_ang < 180):      daseg[i] = sign * (delta_ang)
#                elif (ang2 > ang1) & (delta_ang > 180):    daseg[i] = sign * (delta_ang - 360)
#                elif (ang2 < ang1) & (delta_ang > -180):   daseg[i] = sign * (delta_ang)
#                elif (ang2 < ang1) & (delta_ang < -180):   daseg[i] = sign * (delta_ang + 360)                
#
#        # add to dataframe
#        df_['dpseg'] = dpseg
#        df_['dvseg'] = dvseg
#        df_['dvxseg'] = dvxseg
#        df_['dvyseg'] = dvyseg
#        df_['daseg'] = daseg
#        df_['dlseg'] = dlseg
#        df_['dlaseg'] = dlaseg
#        df_['dwseg'] = dwseg
#        if 'rmw' in df_.keys():   df_['drseg'] = drseg
#
#        # store number nodes
#        num_st_dseg.append(df_.dropna().shape[0])
#        num_st.append(np.unique(df_.dropna().id).size)
#
#        # store (order 'NA','SA','WP','EP','SP','NI','SI')
#        df_center_ls.append(df_)
#        
#    # store nums
#    df_num = pd.DataFrame(
#        {
#        'n_basin': np.array(num_st_basin),
#        'n_mask': np.array(num_st_mask),
#        'n_nonan': np.array(num_st_nonan),
#        'n_ids': np.array(num_st_ids),
#        'n_6h': np.array(num_st_dt),
#        'n_seg': np.array(num_st_segs),
#        'n_dseg': np.array(num_st_dseg),
#        'n_storms': np.array(num_st),
#        },
#        index = dict_basin,
#    )
#    
#    return df_center_ls, df_num
