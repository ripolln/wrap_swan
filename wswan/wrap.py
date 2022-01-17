#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import shutil
import subprocess as sp
import sys

import numpy as np
import xarray as xr

# SWAN STAT LIBRARY
from .io import SwanIO_STAT, SwanIO_NONSTAT


# grid description template
d_grid_template = {
    'xpc': None,      # x origin
    'ypc': None,      # y origin
    'alpc': None,     # x-axis direction 
    'xlenc': None,    # grid length in x
    'ylenc': None,    # grid length in y
    'mxc': None,      # number mesh x
    'myc': None,      # number mesh y
    'dxinp': None,    # size mesh x
    'dyinp': None,    # size mesh y
}

# swan input parameters template
d_params_template = {

    # SET general parameters
    'set_level': None,         # increase in water level (m)
    'set_maxerr': None,        # input data error tolerance: 1 (default), 2, 3
    'set_cdcap': None,         # max value for wind drag coeff (2.5*10^3): None, value
    'set_convention': None,    # wind/waves angle convention: CARTESIAN (default), NAUTICAL

    # COORDINATES
    'coords_mode': None,       # coordinates system 'CARTESIAN' (default), 'SPHERICAL' 
    'coords_projection': None, # projection method: 'CCM' (default), 'QC'

    # COMPUTATIONAL GRID
    'cgrid_mdc': 72,           # spectral circle subdivisions
    'cgrid_flow': 0.03,        # lowest discrete frequency used in the calculation (Hz) (0.03) 
    'cgrid_fhigh': 1.00,       # highest discrete frequency used in the calculation (Hz) (1.00) 

    # BOUNDARY WAVES
    'boundw_jonswap': None,    # jonswap gamma: 3.3 (default)
    'boundw_period': None,     # waves period: 'PEAK', 'MEAN' 

    # BOUNDARY NESTED
    'boundn_mode': 'CLOSED',   # input from main mesh to nested mesh: OPEN, CLOSED (default) 

    # PHYSICS: See (http://swanmodel.sourceforge.net/online_doc/swanuse/node28.html)
    'physics': [],            # list of raw .swn entries

    # NUMERICS: See (http://swanmodel.sourceforge.net/online_doc/swanuse/node29.html)
    'numerics': [],           # list of raw .swn entries

    # -- exclusive NONSTAT parameters--

    # INPUT GRIDs
    'wind_deltinp': None,      # wind input delta time: '5 MIN', '1 HR', ... (us: SEC, MIN, HR, DAY)
    'level_deltinp': None,     # level input delta time: '5 MIN', '1 HR', ... (us: SEC, MIN, HR, DAY)

    # COMPUTE
    'compute_deltc': None,     # computation delta time '5 MIN', '1 HR', ... (us: SEC, MIN, HR, DAY)

    # OUTPUT
    'output_deltt': None,      # output delta time '5 MIN', '1 HR', ... (us: SEC, MIN, HR, DAY)
    'output_variables': [
        'HSIGN', 'DIR', 'PDIR', 'TM02',
        'TPS', 'RTP', 'FSPR', 'DSPR',
        'DEPTH', 'WATLEV', 'WIND'],    # output varibles (compgrid) to be stored

    # OUTPUT storage custom initial time (delta hours from case start)
    'output_time_ini_specout': None,
    'output_time_ini_block': None,
    'output_time_ini_table': None,

    # x,y output points (optional)
    'output_points_x': [],
    'output_points_y': [],
    'output_variables_points': [
        'HSIGN', 'DIR', 'PDIR', 'TM02',
        'TPS', 'RTP', 'FSPR', 'DSPR',
        'DEPTH', 'WATLEV', 'WIND'],    # output varibles (tablepoints) to be stored

    # output spectra (optional)
    'output_spec_deltt': None,    # output delta time '5 MIN', '1 HR', ... (us: SEC, MIN, HR, DAY)
    'output_spec': False,         # activates COMPGRID for spectra storage
    'output_points_spec': False,  # activates OUTPTS for spectra storage
}


class SwanMesh(object):
    'SWAN numerical model mesh'

    def __init__(self):

        self.depth = None   # bathymetry depth value (2D numpy.array)

        self.ID = ''        # mesh ID ("main", "nest1", ...)

        # mesh related filenames
        self.fn_depth = 'depth_{0}.dat'    # filename used in SWAN execution
        self.fn_output = 'output_{0}.mat'  # output .mat file for mesh comp. grid
        self.fn_output_spec = 'spec_compgrid_{0}.dat'       # output spec compgrid file
        self.fn_output_points = 'table_outpts_{0}.dat'      # output points file
        self.fn_output_points_spec = 'spec_outpts_{0}.dat'  # output spec points file
        self.fn_input = 'input_{0}.swn'    # input .swn file

        # for nested mesh
        self.is_nested = False
        self.fn_boundn = 'bounds_{0}.dat'  # input bounds file

        # grid parameters
        self.cg = d_grid_template.copy()  # computational grid
        self.dg = d_grid_template.copy()  # depth grid
        self.dg_idla = 1  # http://swanmodel.sourceforge.net/online_doc/swanuse/node26.html

    def set_ID(self, ID):
        'set mesh ID and related files names'

        self.ID = ID

        self.fn_depth = 'depth_{0}.dat'.format(ID)
        self.fn_output = 'output_{0}.mat'.format(ID)
        self.fn_output_spec = 'spec_compgrid_{0}.dat'.format(ID)
        self.fn_output_points = 'table_outpts_{0}.dat'.format(ID)
        self.fn_output_points_spec = 'spec_outpts_{0}.dat'.format(ID)
        self.fn_input = 'input_{0}.swn'.format(ID)

        self.fn_boundn = 'bounds_{0}.dat'.format(ID)

    def export_depth(self, p_case):
        'exports depth values to .dat file'

        # TODO: only compatible with dg_idla = 1 ?

        p_export = op.join(p_case, self.fn_depth)
        np.savetxt(p_export, self.depth, fmt='%.2f')

    def get_XY(self):
        # TODO switch computational grid / depth grid (cg/dg)
        'returns mesh X, Y arrays from computational grid'

        # computational grid
        cg = self.cg

        x0 = cg['xpc']
        x1 = cg['xlenc'] + cg['xpc'] - cg['dxinp']
        xN = cg['mxc']
        X = np.linspace(x0, x1, xN)

        y0 = cg['ypc']
        y1 = cg['ylenc'] + cg['ypc'] - cg['dyinp']
        yN = cg['myc']
        Y = np.linspace(y0, y1, yN)

        return X, Y


class SwanProject(object):
    'SWAN numerical model project parameters, grids and information'

    def __init__(self, p_proj, n_proj):
        '''
        SWAN project information will be stored here

        http://swanmodel.sourceforge.net/online_doc/swanuse/node25.html
        '''

        self.p_main = op.join(p_proj, n_proj)    # project path
        self.name = n_proj                       # project name

        # sub folders 
        self.p_cases = op.join(self.p_main, 'cases')  # project cases

        # project main mesh and nested meshes
        self.mesh_main = None       # SwanMesh object for main mesh
        self.mesh_nested_list = []  # list of SwanMesh objects for nested meshes

        # swan execution parameters
        self.params = d_params_template.copy()

        # additional data (optional, used at plots)
        self.shore = np.array([])

    def set_main_mesh(self, sm):
        'Set main mesh, sm - SwanMesh object'

        # main mesh ID 
        sm.set_ID('main')
        self.mesh_main = sm

    def set_nested_mesh_list(self, sm_list):
        'Set project nested mesh list, sm_list - SwanMesh objects list'

        l_nested  = []
        for c, sm_n in enumerate(sm_list):

            # nest mesh ID
            sm_n.set_ID('nest{0}'.format(c))
            sm_n.is_nested = True
            l_nested.append(sm_n)
        self.mesh_nested_list = l_nested

    def set_params(self, input_params):
        'Set project parameters from input dictionary'

        # update template parameters 
        self.params = {**self.params, **input_params}


class SwanWrap(object):
    'SWAN numerical model wrap for multi-case handling'

    def __init__(self, swan_proj, swan_io):
        '''
        swan_proj - SwanProject() instance, contains project parameters
        swan_io   - SwanIO_STAT / SwanIO_NONSTAT modules (auto from children)
        '''

        # set project and IO module
        self.proj = swan_proj           # swan project parameters
        self.io = swan_io(self.proj)     # swan input/output 

        # swan bin executable
        p_res = op.join(op.dirname(op.realpath(__file__)), 'resources')
        self.bin = op.abspath(op.join(p_res, 'swan_bin', 'swan_ser.exe'))

    def get_run_folders(self):
        'return sorted list of project cases folders'

        # TODO: will find previously generated cases... fix it 

        ldir = sorted(os.listdir(self.proj.p_cases))
        fp_ldir = [op.join(self.proj.p_cases, c) for c in ldir]

        return [p for p in fp_ldir if op.isdir(p)]

    def run_cases(self):
        'run all cases inside project "cases" folder'

        def save_swan_prints(p_run, mesh_id):
            'copy swan Errfile, norm_end and PRINT files with mesh name'

            for fk in ['Errfile', 'norm_end', 'PRINT']:
                if op.isfile(op.join(p_run, fk)):
                    shutil.copy(
                        op.join(p_run, fk),
                        op.join(p_run, '{0}_{1}'.format(fk, mesh_id))
                    )

        # TODO: improve log / check execution ending status

        # get sorted execution folders
        run_dirs = self.get_run_folders()

        for p_run in run_dirs:

            # run case main mesh
            self.run(p_run, input_file = self.proj.mesh_main.fn_input)
            save_swan_prints(p_run, self.proj.mesh_main.ID )

            # run nested meshes
            for mesh_n in self.proj.mesh_nested_list:
                self.run(p_run, input_file = mesh_n.fn_input)
                save_swan_prints(p_run, mesh_n.ID )

            # log
            p = op.basename(p_run)
            print('SWAN CASE: {0} SOLVED'.format(p))

    def run(self, p_run, input_file='input_main.swn'):
        'Bash execution commands for launching SWAN'

        # aux. func. for launching bash command
        def bash_cmd(str_cmd, out_file=None, err_file=None):
            'Launch bash command using subprocess library'

            _stdout = None
            _stderr = None

            if out_file:
                _stdout = open(out_file, 'w')
            if err_file:
                _stderr = open(err_file, 'w')

            s = sp.Popen(str_cmd, shell=True, stdout=_stdout, stderr=_stderr)
            s.wait()

            if out_file:
                _stdout.flush()
                _stdout.close()
            if err_file:
                _stderr.flush()
                _stderr.close()

        # check if windows OS
        is_win = sys.platform.startswith('win')

        if is_win:
            # WINDOWS - use swashrun command
            cmd = 'cd {0} && copy {1} INPUT && {2} INPUT'.format(
                p_run, input_file, self.bin)

        else:
            # LINUX/MAC - ln input file and run swan case
            cmd = 'cd {0} && ln -sf {1} INPUT && {2} INPUT'.format(
                p_run, input_file, self.bin)

        bash_cmd(cmd)

    def extract_output(self, case_ini=None, case_end=None, mesh=None,
                       var_name=None, concat=True):
        '''
        exctract output from stationary and non stationary cases
        (it is possible to choose which cases to extract)

        var_name option allow for one unique variable extraction

        concat - True to  return xarray.Dataset with dimension "case" added
        else return list of xarray.Dataset (adds "case" value to each output)
        '''

        # select main or nested mesh
        if mesh == None: mesh = self.proj.mesh_main

        # get sorted execution folders
        run_dirs = self.get_run_folders()
        cs_ix = range(0, len(run_dirs))
        if (case_ini != None) & (case_end != None):
            run_dirs = run_dirs[case_ini:case_end]
            cs_ix = range(case_ini, case_end)

        # exctract output case by case and concat in list
        l_out = []
        for c, p_run in zip(cs_ix, run_dirs):

            # read output file
            xds_case_out = self.io.output_case(p_run, mesh)

            # optional chose variable
            if var_name != None:
                xds_case_out = xds_case_out[[var_name]]

            # store case id
            xds_case_out['case'] = c

            l_out.append(xds_case_out)

        if concat:
            return(xr.concat(l_out, dim='case'))

        else:
            return(l_out)


class SwanWrap_STAT(SwanWrap):
    'SWAN numerical model wrap for STATIONARY multi-case handling'

    def __init__(self, swan_proj):
        super().__init__(swan_proj, SwanIO_STAT)

    def build_cases(self, waves_dataset):
        '''
        generates all files needed for swan stationary multi-case execution

        waves_dataset - pandas.dataframe with "n" boundary conditions setup
        [n x 4] (hs, per, dir, spr)
        '''

        # make main project directory
        self.io.make_project()

        # one stat case for each wave sea state
        for ix, (_, ws) in enumerate(waves_dataset.iterrows()):

            # build stat case 
            case_id = '{0:04d}'.format(ix)
            self.io.build_case(case_id, ws)

    def extract_output_points(self, mesh=None):
        '''
        extract output from points all cases table_outpts.dat

        return xarray.Dataset (uses new dim "case" to join output)
        '''

        # TODO: develop SwanIO_STAT.output_points()

        # select main or nested mesh
        if mesh == None: mesh = self.proj.mesh_main

        # get sorted execution folders
        run_dirs = self.get_run_folders()

        # exctract output case by case and concat in list
        l_out = []
        for p_run in run_dirs:

            # read output file
            xds_case_out = self.io.output_points(p_run, mesh)
            l_out.append(xds_case_out)

        # concatenate xarray datasets (new dim: case)
        xds_out = xr.concat(l_out, dim='case')

        return(xds_out)


class SwanWrap_NONSTAT(SwanWrap):
    'SWAN numerical model wrap for NON STATIONARY multi-case handling'

    def __init__(self, swan_proj):
        super().__init__(swan_proj, SwanIO_NONSTAT)

    def build_cases(self, waves_event_list, storm_track_list=None,
                    make_waves=True, make_winds=True, make_levels=True):
        '''
        generates all files needed for swan non-stationary multi-case execution

        waves_event_list - list waves events time series (pandas.DataFrame)
        also contains level, tide and wind (not storm track) variables
        [n x 8] (hs, per, dir, spr, U10, V10, level, tide)

        storm_track_list - list of storm tracks time series (pandas.DataFrame)
        storm_track generated winds have priority over waves_event winds
        [n x 6] (move, vf, lon, lat, pn, p0)
        '''

        # check user input: no storm tracks
        if storm_track_list == None:
            storm_track_list = [None] * len(waves_event_list)

        # make main project directory
        self.io.make_project()

        # one non-stationary case for each wave time series
        for ix, (wds, sds) in enumerate(
            zip(waves_event_list, storm_track_list)):

            # build stat case 
            case_id = '{0:04d}'.format(ix)
            self.io.build_case(
                case_id, wds, storm_track=sds,
                make_waves=make_waves, make_winds=make_winds,
                make_levels=make_levels,
            )

    def extract_output_spec(self, case_ini=None, case_end=None, mesh=None,
                            var_name=None):
        '''
        exctract output from non stationary cases
        (it is possible to choose which cases to extract)

        var_name option allow for one unique variable extraction

        return list of xarray.Dataset (adds "case" value to each output)
        '''

        # TODO: introducir opcional usar netcdf4

        # select main or nested mesh
        if mesh == None: mesh = self.proj.mesh_main

        # get sorted execution folders
        run_dirs = self.get_run_folders()
        cs_ix = range(0, len(run_dirs))
        if (case_ini != None) & (case_end != None):
            run_dirs = run_dirs[case_ini:case_end]
            cs_ix = range(case_ini, case_end)

        # exctract output case by case and concat in list
        l_out = []
        for c, p_run in zip(cs_ix, run_dirs):

            # read output file
            xds_case_out = self.io.output_case_spec(p_run, mesh)

            # optional chose variable
            if var_name != None:
                xds_case_out = xds_case_out[[var_name]]

            # store case id
            xds_case_out['case'] = c

            l_out.append(xds_case_out)

        return(l_out)

    def extract_output_points(self, case_ini=None, case_end=None, mesh=None):
        '''
        extract output from points all cases table_outpts.dat
        (it is possible to choose which cases to extract)

        return xarray.Dataset (uses new dim "case" to join output)
        '''

        # select main or nested mesh
        if mesh == None: mesh = self.proj.mesh_main

        # get sorted execution folders
        run_dirs = self.get_run_folders()
        if (case_ini != None) & (case_end != None):
            run_dirs = run_dirs[case_ini:case_end]

        # exctract output case by case and concat in list
        l_out = []
        for p_run in run_dirs:

            # read output file
            xds_case_out = self.io.output_points(p_run, mesh)
            l_out.append(xds_case_out)

        # concatenate xarray datasets (new dim: case)
        xds_out = xr.concat(l_out, dim='case')
        if (case_ini != None) & (case_end != None):
            xds_out = xds_out.assign(case=np.arange(case_ini, case_end))

        return(xds_out)

    def extract_output_points_spec(self, case_ini=None, case_end=None, mesh=None):
        '''
        extract output spectra from points all cases table_outpts.dat
        (it is possible to choose which cases to extract)

        return xarray.Dataset (uses new dim "case" to join output)
        '''

        # select main or nested mesh
        if mesh == None: mesh = self.proj.mesh_main

        # get sorted execution folders
        run_dirs = self.get_run_folders()
        if (case_ini != None) & (case_end != None):
            run_dirs = run_dirs[case_ini:case_end]

        # exctract output case by case and concat in list
        l_out = []
        for p_run in run_dirs:

            # read output file
            xds_case_out = self.io.output_points_spec(p_run, mesh)
            l_out.append(xds_case_out)

        # concatenate xarray datasets (new dim: case)
        xds_out = xr.concat(l_out, dim='case')
        if (case_ini != None) & (case_end != None):
            xds_out = xds_out.assign(case=np.arange(case_ini, case_end))

        return(xds_out)

