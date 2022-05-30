#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import shutil as su
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat
from scipy import interpolate

from .vortex import vortex_model

# output variables metadata 'CODE':('out_name', 'units', 'description')
meta_out_swn = {
    'HSIGN':  ('Hsig', 'm', 'Significant Wave Height'),
    'DIR':    ('Dir', 'º', 'Waves Direction'),
    'PDIR':   ('PkDir', 'º', 'PkDir'),
    'TM02':   ('Tm02', 's', 'Waves Mean Period'),
    'TPS':    ('Tp', 's', 'Waves Peak Period'),
    'DSPR':   ('Dspr', 'º', 'Waves Directional Spread'),
    'WATLEV': ('WaterLevel', 'm', 'Water Level'),
    'WIND_X': ('Windv_x', 'm/s', 'Wind Speed (x)'),
    'WIND_Y': ('Windv_y', 'm/s', 'Wind Speed (y)'),
    'PTHSIGN':('Hs_part', 'm', 'Partition of Significant Wave Height'),
    'PTRTP':  ('Tp_part', 's', 'Partition of Waves Peak Period'),
    'PTDIR':  ('Dir_part', 'º', 'Partition of Waves Direction'),
    'PTDSPR': ('Dspr_part', 'º', 'Partition of directional spread'),
    'PTWFRAC':('Wfrac_part', '-', 'Partition of wind fraction'),
    'PTWLEN': ('Wlen_part', 'm', 'Partition of average wave length'),
    'PTSTEEP':('Steep_part', '-', 'Partition of wave steepness'),
    'OUT':    ('OUT', '-', 'OUT'),
}

# same but for .mat output keys 
cmat = ['Hsig', 'Dir', 'PkDir', 'Tm02', 'TPsmoo', 'Dspr', 'Watlev', 'Windv_x', 'Windv_y',
        'HsPT', 'TpPT', 'DrPT', 'DsPT', 'WfPT', 'WlPT', 'StPT']
cmet = ['HSIGN', 'DIR', 'PDIR', 'TM02', 'TPS', 'DSPR', 'WATLEV', 'WIND_X', 'WIND_Y',
        'PTHSIGN', 'PTRTP', 'PTDIR', 'PTDSPR', 'PTWFRAC', 'PTWLEN', 'PTSTEEP']
meta_out_mat = {k:meta_out_swn[v] for k,v in zip(cmat,cmet)}


# input.swn TEMPLATES - COMMON

def swn_coordinates(proj):
    'COORDINATES .swn block'

    coords_mode = proj.params['coords_mode']
    coords_projection = proj.params['coords_projection']

    proj_str = ''
    if coords_projection: proj_str = '{0}'.format(coords_projection)

    t = ''
    if coords_mode:
        t = 'COORDINATES {0} {1}\n'.format(coords_mode, proj_str)

    return t

def swn_set(proj):
    'SET .swn block'

    set_level = proj.params['set_level']
    set_maxerr = proj.params['set_maxerr']
    set_cdcap = proj.params['set_cdcap']
    set_convention = proj.params['set_convention']

    level_str = 'level={0}'.format(set_level)
    cdcap_str = ''
    if set_cdcap: cdcap_str = 'cdcap={0}'.format(set_cdcap)
    maxerr_str = ''
    if set_maxerr: maxerr_str = 'maxerr={0}'.format(set_maxerr)
    conv_str = ''
    if set_convention: conv_str = '{0}'.format(set_convention)

    return 'SET {0} {1} {2} {3}\n$\n'.format(
        level_str, cdcap_str, maxerr_str, conv_str,
    )

def swn_computational(proj, mesh):
    'COMPUTATIONAL GRID .swn block'
    # only regular grid and full circle spectral directions !!!

    cgrid_mdc = proj.params['cgrid_mdc']
    cgrid_flow = proj.params['cgrid_flow']
    cgrid_fhigh = proj.params['cgrid_fhigh']

    # TODO: parametro [msc] se deja ausente?
    return 'CGRID REGULAR {0} {1} {2} {3} {4} {5} {6} CIRCLE {7} {8} {9} \n$\n'.format(
        mesh.cg['xpc'], mesh.cg['ypc'], mesh.cg['alpc'],
        mesh.cg['xlenc'], mesh.cg['ylenc'], mesh.cg['mxc']-1, mesh.cg['myc']-1,
        cgrid_mdc, cgrid_flow, cgrid_fhigh,
    )

def swn_bathymetry(mesh):
    'BATHYMETRY GRID .swn block'

    mxc = mesh.dg['mxc']
    myc = mesh.dg['myc']

    # fix nested / main different behaviour
    if mesh.is_nested:
        mxc = mxc - 1
        myc = myc - 1

    t = ''
    t += 'INPGRID BOTTOM REGULAR {0} {1} {2} {3} {4} {5} {6}\n'.format(
        mesh.dg['xpc'], mesh.dg['ypc'], mesh.dg['alpc'],
        mxc, myc, mesh.dg['dxinp'], mesh.dg['dyinp'])

    t += "READINP BOTTOM 1 '{0}' {1} 0 FREE\n$\n".format(
        mesh.fn_depth, mesh.dg_idla)

    return t

def swn_physics(proj):
    'PHYSICS .swn block'

    list_physics = proj.params['physics']
    t = ''
    for l in list_physics: t += '{0}\n'.format(l)
    t += '$\n'
    return t

def swn_numerics(proj):
    'NUMERICS .swn block'

    list_numerics = proj.params['numerics']
    t = ''
    for l in list_numerics: t += '{0}\n'.format(l)
    t += '$\n'
    return t

def swn_bound_waves_nested(proj, boundn_file):
    'boundary waves (NESTED) .swn block'

    boundn_mode = proj.params['boundn_mode']

    t = ''
    t += "BOUN NEST '{0}' {1}\n".format(boundn_file, boundn_mode)
    t += '$\n'

    return t

def swn_nestout(proj, t0_iso=None, compute_deltc=None):
    'output for nested boundary waves .swn block'

    mesh_main = proj.mesh_main

    t = ''
    for mesh_n in proj.mesh_nested_list:

        t += "NGRID '{0}' {1} {2} {3} {4} {5} {6} {7}\n".format(
            mesh_n.ID, mesh_n.cg['xpc'], mesh_n.cg['ypc'], mesh_n.cg['alpc'],
            mesh_n.cg['xlenc'], mesh_n.cg['ylenc'],
            np.int32(mesh_n.cg['xlenc'] / mesh_main.cg['dxinp']),
            np.int32(mesh_n.cg['ylenc'] / mesh_main.cg['dyinp'])
        )

        # prepare nonstat times str
        nonstat_str = ''
        if t0_iso != None:
            nonstat_str = 'OUT {0} {1}'.format(t0_iso, compute_deltc)

        t += "NESTOUT '{0}' '{1}' {2} \n$\n".format(mesh_n.ID, mesh_n.fn_boundn, nonstat_str)

    return t

# input.swn TEMPLATES - STAT

def swn_bound_waves_stat(proj, ws, waves_bnd):
    'boundary waves (STAT) .swn block'

    boundw_jonswap = proj.params['boundw_jonswap']
    boundw_period = proj.params['boundw_period']

    t = ''
    t += 'BOUND SHAPespec JONswap {0} {1} DSPR DEGR\n'.format(
        boundw_jonswap, boundw_period)
    for ic in waves_bnd:
        t += "BOUN SIDE {0} CONstant PAR {1:.3f} {2:.3f} {3:.3f} {4:.3f}\n".format(
            ic, ws.hs, ws.per, ws.dir, ws.spr)
    t += '$\n'

    return t

# input.swn TEMPLATES - NONSTAT

def swn_bound_waves_nonstat(proj, waves_bnd):
    'boundary waves (NONSTAT) .swn block'

    boundw_jonswap = proj.params['boundw_jonswap']
    boundw_period = proj.params['boundw_period']

    t = ''
    t += 'BOUND SHAPespec JONswap {0} {1} DSPR DEGR\n'.format(
        boundw_jonswap, boundw_period)
    for ic in waves_bnd:
        t += "BOUN SIDE {0} CONstant FILE 'series_waves_{0}.dat'\n".format(ic)
    t += '$\n'

    return t

def swn_bound_segment_waves_nonstat(proj, seg_event, code):
    '''
    boundary waves (NONSTAT) .swn block for segment type boundary

    seg_event - dictionary with segment coordinates and spectra parameters
    '''

    # boundary jonswap and period
    boundw_jonswap = proj.params['boundw_jonswap']
    boundw_period = proj.params['boundw_period']

    # segment series filename
    fn_ss = 'series_segment_{0}.dat'.format(code)

    text_XY = ''
    for i in range(len(seg_event['X'])):
        text_XY += '{0} {1} '.format(seg_event['X'][i], seg_event['Y'][i])

    t = ''
    t += 'BOUND SHAPespec JONswap {0} {1} DSPR DEGR\n'.format(
        boundw_jonswap, boundw_period)

#    t += "BOUndspec SEGMent {0} {1} CONstant PAR {2} {3} {4} {5}\n".format(
#            seg_event['mode'], text_XY, 
#            seg_event['hs'], seg_event['per'], seg_event['dir'], seg_event['dd'])

    t += "BOUndspec SEGMent {0} {1} CONstant FILE '{2}'\n".format(
            seg_event['mode'], text_XY, fn_ss)
    t += '$\n'

    return t

def swn_inp_levels_nonstat(proj, mesh, t0_iso, t1_iso):
    'input level files (NONSTAT) .swn block'

    level_deltinp = proj.params['level_deltinp']

    level_fn = 'series_level_{0}.dat'.format(mesh.ID)
    level_idla = 3 # TODO: comprobar archivos level se generan acorde

    t = ''
    t += 'INPGRID  WLEV  REGULAR {0} {1} {2} {3} {4} {5} {6} NONSTAT {7} {8} {9}\n'.format(
        mesh.cg['xpc'], mesh.cg['ypc'], mesh.cg['alpc'],
        mesh.cg['mxc']-1, mesh.cg['myc']-1, mesh.cg['dxinp'], mesh.cg['dyinp'],
        t0_iso, level_deltinp, t1_iso)
    t += "READINP  WLEV 1. SERIES '{0}' {1} 0 FREE\n$\n".format(
        level_fn, level_idla)

    return t

def swn_inp_winds_nonstat(proj, mesh, t0_iso, t1_iso, wind_deltinp):
    'input level files (NONSTAT) .swn block'

    wind_fn = 'series_wind_{0}.dat'.format(mesh.ID)
    wind_idla = 3

    t = ''
    t += 'INPGRID  WIND  REGULAR {0} {1} {2} {3} {4} {5} {6} NONSTAT {7} {8} {9}\n'.format(
        mesh.cg['xpc'], mesh.cg['ypc'], mesh.cg['alpc'],
        mesh.cg['mxc']-1, mesh.cg['myc']-1, mesh.cg['dxinp'], mesh.cg['dyinp'],
        t0_iso, wind_deltinp, t1_iso)
    t += "READINP  WIND 1. SERIES '{0}' {1} 0 FREE\n$\n".format(
        wind_fn, wind_idla)

    return t


# SWAN INPUT/OUTPUT LIBRARY

class SwanIO(object):
    'SWAN numerical model input/output'

    def __init__(self, swan_proj):

        # needs SwanProject 
        self.proj = swan_proj

    def make_project(self):
        'makes swan project folder and subfolders'

        if not op.isdir(self.proj.p_main): os.makedirs(self.proj.p_main)
        if not op.isdir(self.proj.p_cases): os.makedirs(self.proj.p_cases)

    def output_case(self, p_case, mesh):
        'read .mat output file from non-stationary and returns xarray.Dataset'

        # extract output from selected mesh
        p_mat = op.join(p_case, mesh.fn_output)
        xds_out = self.outmat2xr(p_mat)

        # set X and Y values
        X, Y = mesh.get_XY()
        xds_out = xds_out.assign_coords(X=X)
        xds_out = xds_out.assign_coords(Y=Y)

        # rename to longitude latitude in spherical coords cases
        coords_mode = self.proj.params['coords_mode']
        if coords_mode == 'SPHERICAL':
            xds_out = xds_out.rename({'X':'lon', 'Y':'lat'})

        return xds_out

    def add_metadata(self, xds, meta):
        'Adds metadata (if available) to output xarray.Datasets'

        for vn in xds.variables:
            if vn in meta.keys():
                name, units, descr = meta[vn]
            else:
                name, units, descr = vn, '-', '-'

            xds[vn].attrs.update({'units':units, 'description':descr})
            xds = xds.rename_vars({vn:name})

        return xds

    def fix_partition_vars(self, xds_out):
        'read partitions variables from output .mat file and returns xarray.Dataset'

        # partition variables keys
        partition_keys = ['HsPT', 'TpPT', 'DrPT', 'DsPT', 'WfPT', 'WlPT', 'StPT']

        # output variables
        vns = xds_out.variables

        for k in partition_keys:
            vps = sorted([v for v in vns if v.startswith(k)])

            # skip if no partition key present
            if vps == []: continue

            cc = xr.concat([xds_out[v] for v in vps], 'partition',)

            # drop splitted partition vars and add full partition var
            for v in vps: xds_out = xds_out.drop_vars(v)
            xds_out[k] = cc

        return xds_out


class SwanIO_STAT(SwanIO):
    'SWAN numerical model input/output - STATIONARY cases'

    def make_input(self, p_file, id_run,
                   mesh, ws,
                   waves_bnd=['N', 'E', 'W', 'S'],
                   ttl_run = ''):
        '''
        Writes input.swn file from waves sea state for stationary execution

        p_file      - input.swn file path
        mesh        - SwanMesh instance
        ws          - wave sea state (hs, per, dr, spr)
        waves_bnd   - wave sea state active boundaries

        ttl_run     - project title (optional)

        more info: http://swanmodel.sourceforge.net/online_doc/swanuse/node23.html
        '''

        # -- PROJECT --
        t = "PROJ '{0}' '{1}' '{2}'\n$\n".format(self.proj.name, id_run, ttl_run)

        # -- MODE STATIONARY --
        t += 'MODE STAT\n'

        # -- COORDINATES --
        t += swn_coordinates(self.proj)

        # -- SET -- 
        t += swn_set(self.proj)

        # -- COMPUTATIONAL GRID --
        t += swn_computational(self.proj, mesh)

        # -- BATHYMETRY --
        t += swn_bathymetry(mesh)

        # -- SWAN STATIONARY -- INPUT WAVES --

        # MAIN mesh - boundary waves
        if not mesh.is_nested:
            t += swn_bound_waves_stat(self.proj, ws, waves_bnd)

        # NESTED mesh - nested waves
        else:
            t += swn_bound_waves_nested(self.proj, mesh.fn_boundn)

        # -- PHYSICS --
        t += swn_physics(self.proj)

        # -- NUMERICS --
        t += swn_numerics(self.proj)

        # -- OUTPUT: NESTED MESHES  -- 
        if not mesh.is_nested:
            t += swn_nestout(self.proj)

        # output variables
        out_vars = ' '.join(self.proj.params['output_variables'])

        # -- OUTPUT: BLOCK  -- 
        t += "BLOCK 'COMPGRID' NOHEAD '{0}' LAY 3 {1}\n$\n".format(
            mesh.fn_output, out_vars)

        # TODO: add SPECOUT line
        # SPECOUT 'COMPGRID' SPEC2D ABS 'outputspec_filename'

        # -- COMPUTE --
        t += 'TEST  1,0\n'
        t += 'COMPUTE \n'
        t += 'STOP\n$\n'

        # write file:
        with open(p_file, 'w') as f:
            f.write(t)

        # log    
        fmt2 = ' 7.2f'
        print(
            'SWAN CASE: {1} ---> hs {2:{0}}, per {3:{0}}, dir {4:{0}}, spr {5:{0}}'.format(
                fmt2, id_run, ws.hs, ws.per, ws.dir, ws.spr
            )
        )

    def build_case(self, case_id, waves_ss, waves_bnd=['N', 'E', 'W', 'S']):
        '''
        Build SWAN STAT case input files for given wave sea state (hs, per, dir, spr)

        ix_case  - SWAN case index (int)
        waves_ss - wave sea state (hs, per, dir, spr)
        bnd      - wave sea state active boundaries
        '''

        # SWAN case path
        p_case = op.join(self.proj.p_cases, case_id)
        if not op.isdir(p_case): os.makedirs(p_case)

        # MAIN mesh
        self.proj.mesh_main.export_depth(p_case)  # export main depth file
        p_swn = op.join(p_case, self.proj.mesh_main.fn_input)

        # make input.swn file
        self.make_input(
            p_swn, case_id,
            self.proj.mesh_main,
            waves_ss,
            waves_bnd = waves_bnd,
        )

        # NESTED meshes 
        for mesh_nested in self.proj.mesh_nested_list:

            mesh_nested.export_depth(p_case)  # export nested depth file
            p_swn = op.join(p_case, mesh_nested.fn_input)

            # make input_nestX.swn file
            self.make_input(
                p_swn, case_id,
                mesh_nested,
                waves_ss,
            )

    def outmat2xr(self, p_mat):
        'read output .mat file and returns xarray.Dataset'

        # matlab dictionary
        dmat = loadmat(p_mat)

        # find output variable keys inside .mat file
        ks = list(set([x.split('_')[0] for x in dmat.keys()]))
        ks = [x for x in ks if x]  # remove empty values
        if 'Windv' in ks: ks.remove('Windv'); ks.append('Windv_x'); ks.append('Windv_y')

        # iterate  variables 
        xds_out = xr.Dataset()
        for vn in ks:
            xds_out[vn] = (('Y','X',), dmat['{0}'.format(vn)])

        # join partitions variables (if any)
        xds_out = self.fix_partition_vars(xds_out)

        # add variable metadata (if available)
        xds_out = self.add_metadata(xds_out, meta_out_mat)

        return xds_out

    def output_points(self, p_case, mesh):
        'read table_outpts_meshID.dat output file and returns xarray.Dataset'

        # TODO def read output POINTS for STAT cases

        return None

    def read_outpts_spec(self, p_outpts_spec):
        'Read output spectral data for swan SPECOUT text file'

        # TODO: def read output SPEC for STAT cases
        # (same as at SwanIO_NONSTAT but without TIME)

        return None


class SwanIO_NONSTAT(SwanIO):
    'SWAN numerical model input/output - NON STATIONARY cases'

    def make_out_points(self, p_file):
        'Generates desired output-points coordinates file'

        # define and save output points
        x_out = self.proj.params['output_points_x']
        y_out = self.proj.params['output_points_y']

        if not x_out or not y_out:
            return

        else:
            points = np.vstack((x_out,y_out)).T
            np.savetxt(p_file, points, fmt='%.2f')

    def make_wave_boundary_files(self, p_case, waves_event, time, bnd):
        'Generate event wave files (swan compatible)'

        # wave variables
        hs = waves_event.hs.values[:]
        per = waves_event.per.values[:]
        direc = waves_event.dir.values[:]
        spr = waves_event.spr.values[:]

        # csv file 
        num_data = len(time)
        data = np.zeros((num_data, 5))
        data[:, 0] = time
        data[:, 1] = hs
        data[:, 2] = per
        data[:, 3] = direc
        data[:, 4] = spr

        # Copy file for all boundaries
        save = op.join(p_case, 'series_waves.dat')
        np.savetxt(save, data, header='TPAR', comments='', fmt='%8.4f %2.3f %2.3f %3.2f %3.1f')
        for i in bnd:
            su.copyfile(save, op.join(p_case, 'series_waves_{0}.dat'.format(i)))

    def make_wave_segment_files(self, p_case, segment, time, code):
        'Generate event wave files for segment boundary input (swan compatible)'

        # TODO: esta entrando "time_swan" pero el evento ya lleva su propio
        # time

        # wave variables
        hs = segment['waves_event'].hs.values[:]
        per = segment['waves_event'].per.values[:]
        direc = segment['waves_event'].dir.values[:]
        spr = segment['waves_event'].spr.values[:]

        # csv file 
        num_data = len(time)
        data = np.zeros((num_data, 5))
        data[:, 0] = time
        data[:, 1] = hs
        data[:, 2] = per
        data[:, 3] = direc
        data[:, 4] = spr

        # Copy file for all boundaries
        save = op.join(p_case, 'series_segment_{0}.dat'.format(code))
        np.savetxt(save, data, header='TPAR', comments='', fmt='%8.4f %2.3f %2.3f %3.2f %3.1f')

    def make_wind_files_uniform(self, p_case, wind_series, mesh):
        '''
        Generate event wind mesh files (swan compatible)

        uses wind_series U10 and V10 values at the entire SWAN comp. grid
        '''

        # wind variables
        u10 = wind_series.U10.values[:]
        v10 = wind_series.V10.values[:]

        # each time needs 2D (mesh) wind files (U,V) 
        mxc = mesh.cg['mxc']  # number mesh x
        myc = mesh.cg['myc']  # number mesh y
        code = 'wind_{0}'.format(mesh.ID)

        txt = ''
        for c, (u, v) in enumerate(zip(u10,v10)):

            # single point wind -> entire SWAN comp.grid wind
            aux = np.ones((myc, mxc))

            # TODO: wind has to be rotated if alpc != 0

            # csv file 
            u_2d = aux * u
            v_2d = aux * v

            u_v_stack = np.vstack((u_2d, v_2d))
            save = op.join(p_case, '{0}_{1:06}.dat'.format(code, c))
            np.savetxt(save, u_v_stack, fmt='%.2f')

            # wind list file
            txt += '{0}_{1:06}.dat\n'.format(code, c)

        # winds file path
        save = op.join(p_case, 'series_{0}.dat'.format(code))
        with open(save, 'w') as f:
            f.write(txt)

    def make_wind_files_mesh(self, p_case, wind_2d, mesh):
        '''
        Generate event wind mesh files (swan compatible)

        uses user given wind 2d configuration
        (xarray.Dataset vars: U10, V10. coords: x, y, time)
        '''

        # wind code
        code = 'wind_{0}'.format(mesh.ID)

        txt = ''
        for c, _ in enumerate(wind_2d.time):

            # TODO: wind has to be rotated if alpc != 0

            # get wind u and v data
            u_2d = wind_2d.isel(time=c).U10.values[:]
            v_2d = wind_2d.isel(time=c).V10.values[:]

            # csv file
            u_v_stack = np.vstack((u_2d, v_2d))
            save = op.join(p_case, '{0}_{1:06}.dat'.format(code, c))
            np.savetxt(save, u_v_stack, fmt='%.2f')

            # wind list file
            txt += '{0}_{1:06}.dat\n'.format(code, c)

        # winds file path
        save = op.join(p_case, 'series_{0}.dat'.format(code))
        with open(save, 'w') as f:
            f.write(txt)

    def make_vortex_files(self, p_case, case_id, mesh, storm_track):
        '''
        Generate event vortex wind mesh files (swan compatible)

        mesh        - mesh (main or nested)
        storm track - pandas.Dataframe: move, vf, vfx, vfy, pn, p0, lon, lat, vmax
        '''

        code = 'wind_{0}'.format(mesh.ID)

        # vortex model
        xds_vortex = vortex_model(
            storm_track,
            mesh,
            self.proj.params['coords_mode']
        )

        # each time needs 2D (mesh) wind files (U,V)
        txt = ''
        for c, t in enumerate(xds_vortex.time):
            vortex_t = xds_vortex.isel(time=c)

            # get wind module and dir from vortex
            W = vortex_t.W.values[:]
            ang = np.deg2rad(270 - vortex_t.Dir.values[:])

            # 2D wind field
            u_2d = W * np.cos(ang)  # m/s
            v_2d = W * np.sin(ang)  # m/s
            u_v_stack = np.vstack((u_2d, v_2d))

            # csv file 
            save = op.join(p_case, '{0}_{1:06}.dat'.format(code, c))
            np.savetxt(save, u_v_stack, fmt='%.2f')

            # wind list file
            txt += '{0}_{1:06}.dat\n'.format(code, c)

        # winds file path
        save = op.join(p_case, 'series_{0}.dat'.format(code))
        with open(save, 'w') as f:
            f.write(txt)

        # vortex .nc file
        p_vortex = op.join(p_case, 'vortex_{0}.nc'.format(code))
        xds_vortex.to_netcdf(p_vortex)

    def make_level_files(self, p_case, water_level, mesh):
        'Generate event level mesh files (swan compatible)'

        # parse pandas time index to swan iso format
        swan_iso_fmt = '%Y%m%d.%H%M'
        time = pd.to_datetime(water_level.index).strftime(swan_iso_fmt).values[:]

        # level variables
        zeta = water_level.level.values[:]
        tide = water_level.tide.values[:]

        # each time needs 2D (mesh) level 
        mxc = mesh.cg['mxc']  # number mesh x
        myc = mesh.cg['myc']  # number mesh y
        code = 'level_{0}'.format(mesh.ID)

        txt = ''
        for c, (z, t) in enumerate(zip(zeta, tide)):

            # single point level -> entire SWAN comp.grid level
            aux = np.ones((mxc, myc)).T

            # csv file 
            l = z + t  # total level
            l_2d = aux * l
            save = op.join(p_case, '{0}_{1:06}.dat'.format(code, c))
            np.savetxt(save, l_2d, fmt='%.2f')

            # level list file
            txt += '{0}_{1:06}.dat\n'.format(code, c)

        # level file path
        save = op.join(p_case, 'series_{0}.dat'.format(code))
        with open(save, 'w') as f:
            f.write(txt)

    def calculate_t0_out(self, t0_iso):
        '''
        gets initial output storage for specout, block and table output modes

        t0_iso - initial computation time (SWAN iso format)
        '''

        # default output storage initial time (case start)
        t0_out_spec = t0_iso
        t0_out_block = t0_iso
        t0_out_table = t0_iso

        # custom output storage initial time
        swan_iso_fmt = '%Y%m%d.%H%M'
        t0_dt = datetime.strptime(t0_iso, swan_iso_fmt)

        if self.proj.params['output_time_ini_specout']:
            tdt = t0_dt + timedelta(hours=self.proj.params['output_time_ini_specout'])
            t0_out_spec = tdt.strftime(swan_iso_fmt)

        if self.proj.params['output_time_ini_block']:
            tdt = t0_dt + timedelta(hours=self.proj.params['output_time_ini_block'])
            t0_out_block = tdt.strftime(swan_iso_fmt)

        if self.proj.params['output_time_ini_table']:
            tdt = t0_dt + timedelta(hours=self.proj.params['output_time_ini_table'])
            t0_out_table = tdt.strftime(swan_iso_fmt)

        return t0_out_spec, t0_out_block, t0_out_table

    def make_input(self, p_file, id_run,
                   mesh,
                   time, compute_deltc, wind_deltinp=None,
                   ttl_run='',
                   make_waves=True, make_winds=True, make_levels=True,
                   waves_mode='boundary',
                   waves_bnd=['N', 'E', 'W', 'S'],
                   waves_segments=[],
                  ):
        '''
        Writes input.swn file from waves event for non-stationary execution

        p_file     - input.swn file path
        time       - event time at swan iso format
        compute_deltc - computational delta time (swan project parameter)

        ttl_run    - execution title that will appear in the output

        make_waves - activates waves input files generation (at waves_bnd)
        make_winds - activates wind input files generation
        make_levels - activates level input files generation

        more info: http://swanmodel.sourceforge.net/online_doc/swanuse/node23.html
        '''

        # -- PROJECT --
        t = "PROJ '{0}' '{1}' '{2}'\n$\n".format(self.proj.name, id_run, ttl_run)

        # -- MODE NONSTATIONARY --
        t += 'MODE NONSTAT\n'

        # -- COORDINATES --
        t += swn_coordinates(self.proj)

        # -- SET -- 
        t += swn_set(self.proj)

        # -- COMPUTATIONAL GRID --
        t += swn_computational(self.proj, mesh)

        # -- BATHYMETRY --
        t += swn_bathymetry(mesh)

        # -- SWAN NON STATIONARY -- INPUT GRIDS --
        t0_iso = time[0]   # initial time (SWAN ISOFORMAT)
        t1_iso = time[-1]  # end time (SWAN ISOFORMAT)

        # output storage initial times
        t0_out_spec, t0_out_block, t0_out_table = self.calculate_t0_out(t0_iso)

        # level series files
        if make_levels:
            t += swn_inp_levels_nonstat(self.proj, mesh, t0_iso, t1_iso)

        # wind series files
        if make_winds:
            t += swn_inp_winds_nonstat(self.proj, mesh, t0_iso, t1_iso, wind_deltinp)

        # -- BOUNDARY WAVES CONDITIONS --

        # MAIN mesh - boundary waves
        if not mesh.is_nested:
            if make_waves:
                if waves_mode == 'boundary':
                    t += swn_bound_waves_nonstat(self.proj, waves_bnd)
                elif waves_mode == 'segments':
                    for c, segm in enumerate(waves_segments):
                        code = '{0:04d}'.format(c)
                        t += swn_bound_segment_waves_nonstat(self.proj, segm, code)

        # NESTED mesh - nested waves
        else:
            t += swn_bound_waves_nested(self.proj, mesh.fn_boundn)

        # -- PHYSICS --
        t += swn_physics(self.proj)

        # -- NUMERICS --
        t += swn_numerics(self.proj)

        # -- OUTPUT: NESTED MESHES  -- 
        if not mesh.is_nested:
            t += swn_nestout(self.proj, t0_iso=t0_iso, compute_deltc=compute_deltc)

        # output variables
        out_vars = ' '.join(self.proj.params['output_variables'])

        # -- OUTPUT: BLOCK  -- 
        dt_out = self.proj.params['output_deltt']
        t += "BLOCK 'COMPGRID' NOHEAD '{0}' LAY 3 {1} OUT {2} {3}\n$\n".format(
            mesh.fn_output, out_vars, t0_out_block, dt_out)

        # wave spectra at output computational grid 
        if self.proj.params['output_spec']:
            dt_spec_out = self.proj.params['output_spec_deltt']
            t += "SPECOUT 'COMPGRID' SPEC2D ABS '{0}' OUT {1} {2}\n".format(
                mesh.fn_output_spec, t0_out_spec, dt_spec_out)
        t += "$\n"

        # -- OUTPUT: POINTS  -- 
        x_out = self.proj.params['output_points_x']
        y_out = self.proj.params['output_points_y']
        out_vars_table = ' '.join(self.proj.params['output_variables_points'])

        if not x_out or not y_out:
            pass
        else:
            t += "POINTS 'outpts' FILE 'points_out.dat'\n"
            t += "TABLE 'outpts' NOHEAD '{0}' {1} OUT {2} {3}\n".format(
                mesh.fn_output_points, out_vars_table, t0_out_table, dt_out)

            # wave spectra at output points
            if self.proj.params['output_points_spec']:
                t += "SPECOUT 'outpts' SPEC2D ABS '{0}' OUT {1} {2}\n".format(
                    mesh.fn_output_points_spec, t0_out_table, dt_out)
            t += "$\n"

        # -- COMPUTE --
        t += 'TEST  1,0\n'
        t += 'COMPUTE NONSTAT {0} {1} {2}\n'.format(t0_iso, compute_deltc, t1_iso)
        t += 'STOP\n$\n'

        # write file:
        with open(p_file, 'w') as f:
            f.write(t)

    def get_time_swan(self, swan_input):
        '''
        gets time array from swan_input (waves, wind, level)

        parse input time to swan iso format
        '''

        # TODO puede que esta funcion no sea necesaria, repasar los tiempos de
        # oleaje, viento, level, y del archivo input.swn

        # swan iso format
        swan_iso_fmt = '%Y%m%d.%H%M'

        # get time from swan input (priority order: waves > winds > levels)
        if swan_input.waves_activate:
            if swan_input.waves_mode == 'boundary':
                time_base = swan_input.waves_series.index
            elif swan_input.waves_mode == 'segments':
                time_base = swan_input.waves_segments[0]['waves_event'].index

        elif swan_input.wind_mode != None:
            if isinstance(swan_input.wind_series, pd.DataFrame):
                time_base = swan_input.wind_series.index
            elif isinstance(swan_input.wind_series, xr.Dataset):
                time_base = swan_input.wind_series.time.values[:]

        elif swan_input.level_activate:
            time_base = swan_input.level_series.index

        # parse pandas time index to swan iso format
        time_swan = pd.to_datetime(time_base).strftime(swan_iso_fmt).values[:]

        return time_swan

    def build_case(self, case_id, swan_input):
        '''
        Build SWAN NONSTAT case input files for given wave dataset

        case_id     - SWAN case index (int)
        swan_input  - SwanInput_NONSTAT instance
        '''

        # parse pandas time index to swan iso format (get time from input waves series)
        time_swan = self.get_time_swan(swan_input)
        # TODO revisar este time_swan, puede que este ligado solo al oleaje y
        # no haga falta aqui

        # project computational and winds_input delta time
        compute_deltc = self.proj.params['compute_deltc']
        wind_deltinp = self.proj.params['wind_deltinp']

        # SWAN case path
        p_case = op.join(self.proj.p_cases, case_id)
        if not op.isdir(p_case): os.makedirs(p_case)

        # MAIN mesh
        self.proj.mesh_main.export_depth(p_case)  # export depth file

        # make water level files
        if swan_input.level_activate:
            self.make_level_files(p_case, swan_input.level_series, self.proj.mesh_main)

        # make wave files
        if swan_input.waves_activate:

            # boundary waves
            if swan_input.waves_mode == 'boundary':
                self.make_wave_boundary_files(
                    p_case,
                    swan_input.waves_series,
                    time_swan,
                    swan_input.waves_boundaries,
                )

            # segment waves
            elif swan_input.waves_mode == 'segments':
                for c, segment in enumerate(swan_input.waves_segments):
                    # TODO: revisar "time_swan". puede no ser necesario aqui
                    self.make_wave_segment_files(
                        p_case,
                        segment,
                        time_swan,
                        code = '{0:04d}'.format(c),
                    )

        # wind switch
        wind_activate = False
        if swan_input.wind_mode != None:
            wind_activate = True

        # make wind files
        if wind_activate:

            # vortex model from storm tracks  //  meshgrid wind
            if swan_input.wind_mode == 'storm':
                self.make_vortex_files(
                    p_case, case_id,
                    self.proj.mesh_main,
                    swan_input.wind_series,
                )

                # optional: override computational/winds dt with storm track attribute 
                if 'override_dtcomp' in swan_input.wind_series.attrs:
                    compute_deltc = swan_input.wind_series.attrs['override_dtcomp']
                    wind_deltinp = swan_input.wind_series.attrs['override_dtcomp']
                    print('CASE {0} - compute_deltc, wind_deltinp override with storm track: {1}'.format(
                        case_id, compute_deltc))

            # 2d winds given by user
            elif swan_input.wind_mode == '2D':
                self.make_wind_files_mesh(p_case, swan_input.wind_series, self.proj.mesh_main)

            # extrapolate waves event U10, V10 to entire mesh
            elif swan_input.wind_mode == 'uniform':
                self.make_wind_files_uniform(p_case, swan_input.wind_series, self.proj.mesh_main)

        # make output points file
        self.make_out_points(op.join(p_case, 'points_out.dat'))

        # make input.swn file
        p_swn = op.join(p_case, self.proj.mesh_main.fn_input)

        self.make_input(
            p_swn, case_id,
            self.proj.mesh_main,
            time_swan,
            compute_deltc, wind_deltinp,
            make_waves = swan_input.waves_activate,
            waves_mode = swan_input.waves_mode,
            make_winds = wind_activate,
            make_levels = swan_input.level_activate,
            waves_bnd = swan_input.waves_boundaries,
            waves_segments = swan_input.waves_segments,
        )

        # NESTED mesh depth and input (wind, level) files
        for mesh_n in self.proj.mesh_nested_list:

            mesh_n.export_depth(p_case)  # export nested depth file
            p_swn = op.join(p_case, mesh_n.fn_input)

            if swan_input.level_activate:
                self.make_level_files(p_case, swan_input.level_series, mesh_n)

            if wind_activate:

                # vortex model from storm tracks  //  meshgrid wind
                if swan_input.wind_mode == 'storm':
                    self.make_vortex_files(
                        p_case, case_id,
                        mesh_n,
                        swan_input.wind_series,
                    )

                # 2d winds given by user
                elif swan_input.wind_mode == '2D':
                    # TODO for nested mesh we have to adjust wind_series area to nested mesh
                    print('wind_2d + nested mesh not implemented')
                    # TODO THIS WILL FAIL
                    wind_series_nested = swan_input.wind_series

                    self.make_wind_files_mesh(p_case, wind_series_nested, mesh_n)

                # extrapolate waves event U10, V10 to entire mesh
                elif swan_input.wind_mode == 'uniform':
                    self.make_wind_files_uniform(p_case, swan_input.wind_series, mesh_n)

            # make input_nestX.swn file
            self.make_input(
                p_swn, case_id,
                mesh_n,
                time_swan,
                compute_deltc, wind_deltinp,
                make_waves = swan_input.waves_activate,
                make_winds = wind_activate,
                make_levels = swan_input.level_activate,
            )

    def outmat2xr(self, p_mat):
        'read output .mat file and returns xarray.Dataset'

        # matlab dictionary
        dmat = loadmat(p_mat)

        # find output variable keys inside .mat file
        ks = list(set([x.split('_')[0] for x in dmat.keys()]))
        ks = [x for x in ks if x]  # remove empty values
        if 'Windv' in ks: ks.remove('Windv'); ks.append('Windv_x'); ks.append('Windv_y')

        # get dates from one key
        hsfs = sorted([x for x in dmat.keys() if ks[0] == x.split('_')[0]])
        dates_str = ['_'.join(x.split('_')[1:]) for x in hsfs]
        dates = [datetime.strptime(s,'%Y%m%d_%H%M%S') for s in dates_str]

        # iterate times and variables 
        l_times = []
        for ix_t, ds in enumerate(dates_str):
            xds_t = xr.Dataset()
            for vn in ks:
                xds_t[vn] = (('Y','X',), dmat['{0}_{1}'.format(vn, ds)])
            l_times.append(xds_t)

        # join at times dim
        xds_out = xr.concat(l_times, dim='time')
        xds_out = xds_out.assign_coords(time=dates)

        # join partitions variables (if any)
        xds_out = self.fix_partition_vars(xds_out)

        # add variable metadata (if available)
        xds_out = self.add_metadata(xds_out, meta_out_mat)

        return xds_out

    def get_swn_info(self, p_input):
        'get raw information from input.swn file'

        # read entire input.swn
        with open(p_input, 'r') as fR:
            ls = fR.readlines()

        # locate TABLE line and extract some data
        lx = [x for x in ls if x.startswith('TABLE')][0].split(' ')

        # read start date and delta time
        t0_str = lx[-3]  # start date
        swan_iso_fmt = '%Y%m%d.%H%M'
        t0_dt = datetime.strptime(t0_str, swan_iso_fmt)
        dt_min = lx[-2]  # dt (minutes)  # TODO: make compatible user def. UNITS

        # read TABLE output variables codes
        vs_out = lx[4:-3]

        # fix WIND (2 column for x,y)
        if 'WIND' in vs_out:
            ix_i = vs_out.index('WIND'); vs_out.remove('WIND')
            vs_out.insert(ix_i, 'WIND_X')
            vs_out.insert(ix_i+1, 'WIND_Y')

        # return everything in one dict
        swn_info = {
            'TABLE_dtmin': dt_min,
            'TABLE_t0dt': t0_dt,
            'TABLE_vars': vs_out,
        }

        return swn_info

    def output_points(self, p_case, mesh):
        'read table_outpts_meshID.dat output file and returns xarray.Dataset'

        # read some information from .swn file
        p_swn = op.join(p_case, mesh.fn_input)
        swn_info = self.get_swn_info(p_swn)

        # extract output from selected mesh
        p_dat = op.join(p_case, mesh.fn_output_points)

        # variables to extract
        names = swn_info['TABLE_vars']

        # TODO: revisar
        if 'OUT' in names: names.remove('OUT')

        x_out = self.proj.params['output_points_x']
        y_out = self.proj.params['output_points_y']

        # points are mixed at output file
        np_pts = np.genfromtxt(p_dat)
        n_rows = np_pts.shape[0]

        # number of points
        n_pts = len(x_out)

        l_xds_pts = []
        for i in range(n_pts):
            ix_p = np.arange(i, n_rows, n_pts)

            np_pti = np_pts[ix_p, :]
            xds_pti = xr.Dataset({})
            for c, n in enumerate(names):
                xds_pti[n] = (('time'), np_pti[:,c])
            l_xds_pts.append(xds_pti)

        xds_out = xr.concat(l_xds_pts, dim='point')

        # add variable metadata (if available)
        xds_out = self.add_metadata(xds_out, meta_out_swn)

        # add point x and y
        xds_out['x_point'] = (('point'), x_out)
        xds_out['y_point'] = (('point'), y_out)

        # mesh ID
        xds_out.attrs['mesh_ID'] = mesh.ID

        # add times dim values
        t0, dt_min = swn_info['TABLE_t0dt'], swn_info['TABLE_dtmin']
        time_out = pd.date_range(t0, periods=len(xds_out.time), freq='{0}min'.format(dt_min))
        xds_out = xds_out.assign_coords(time=time_out)

        return xds_out

    def read_outpts_spec(self, p_outpts_spec):
        'Read output spectral data for swan SPECOUT text file'

        # TODO puede romper por memoria, usar la alternativa netcdf4

        with open(p_outpts_spec) as f:

            # skip header
            for i in range(3): f.readline()

            # skip TIME lines
            for i in range(2): f.readline()

            # skip LONLAT line
            f.readline()

            # get number of points and coordinates
            n_points = int(f.readline().split()[0])
            lon_points, lat_points = [], []
            for i in range(n_points):
                lonlat = f.readline()
                lon_points.append(float(lonlat.split()[0]))
                lat_points.append(float(lonlat.split()[1]))

            # skip AFREQ line
            f.readline()

            # get FREQs
            n_freq = int(f.readline().split()[0])
            freqs = []
            for i in range(n_freq):
                freqs.append(float(f.readline()))

            # skip NDIR line
            f.readline()

            # get DIRs
            n_dir = int(f.readline().split()[0])
            dirs = []
            for i in range(n_dir):
                dirs.append(float(f.readline()))

            # skip QUANT lines
            for i in range(2): f.readline()

            # skip vadens lines
            for i in range(2): f.readline()

            # get exception value
            ex = float(f.readline().split()[0])

            # start reading spectral output data
            times = []
            specs = []

            # first time line
            tl = f.readline()

            while tl:
                time = datetime.strptime(tl.split()[0], '%Y%m%d.%H%M%S')
                times.append(time)

                # spectral output numpy storage
                spec_pts_t = np.ones((n_freq, n_dir, n_points)) * np.nan

                # read all points
                for p in range(n_points):

                    # check we have result
                    fac_line = f.readline()

                    # read spectra values 
                    if 'FACTOR' in fac_line:

                        # get factor
                        fac = float(f.readline())

                        # read matrix
                        for i in range(n_freq):
                            spec_pts_t[i,:,p] = np.fromstring(f.readline(), sep=' ')

                        # multiply spec by factor
                        spec_pts_t[:,:,p] = spec_pts_t[:,:,p] * fac

                # append spectra
                specs.append(spec_pts_t)

                # read next time line (if any)
                tl = f.readline()

            # file end, stack spec_pts for all times
            spec_out = np.stack(specs, axis=-1)

            # mount output to xarray.Dataset 
            return xr.Dataset(
                {
                    'lon_pts': (('point',), lon_points),
                    'lat_pts': (('point',), lat_points),
                    'spec': (('frequency', 'direction', 'point', 'time'), spec_out),
                },
                coords = {
                    "frequency": freqs,
                    "direction": dirs,
                    "time": times,
                }
            )

    def output_points_spec(self, p_case, mesh):
        'read spec_outpts_meshID.dat output file and returns xarray.Dataset'

        # extract output from selected mesh
        p_dat = op.join(p_case, mesh.fn_output_points_spec)

        # parse spec_output to xarray
        xds_out = self.read_outpts_spec(p_dat)

        # mesh ID
        xds_out.attrs['mesh_ID'] = mesh.ID

        return xds_out

    def output_case_spec(self, p_case, mesh):
        'read spec_compgrid_meshID.dat output file and returns xarray.Dataset'

        # TODO: cuando SwanIO_STAT() tenga su "read_outpts_spec" mover esta funcion a SwanIO()

        # extract output from selected mesh
        p_dat = op.join(p_case, mesh.fn_output_spec)

        # parse spec_output to xarray
        xds_out = self.read_outpts_spec(p_dat)

        # mesh ID
        xds_out.attrs['mesh_ID'] = mesh.ID

        return xds_out

