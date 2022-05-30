# wrap SWAN 

SWAN numerical model python wrapper

## Description

WSWAN toolbox can be used to build and launch SWAN numerical model cases from a Python environment, and to extract model output once execution ends. 

Stationary cases are built from input waves and wind static conditions.

Non-stationary cases require a time series of wave conditions. Additional methodologies are included to build cases from storm tracks, using vortex numerical model for wind fields generation.

An alternative methodology for solving non-stationary wind cases by splitting storm tracks into segments is included in the stopmotion submodule. 

A plotting toolbox is included for case input and output visualization.

## Main contents

Modules included 

[wswan](./wswan/): SWAN numerical model toolbox 
- [io](./wswan/io.py): SWAN numerical model input/output operations
- [wrap](./wswan/wrap.py): SWAN numerical model python wrap 
- [geo](./wswan/geo.py): azimuth distance function
- [stopmotion](./wswan/stopmotion.py): stopmotion methodology module
- [storms](./wswan/storms.py): storm parameters function 
- [vortex](./wswan/vortex.py): vortex winds model 
- [plots](./wswan/plots/): plotting module 
  - [plots - common](./wswan/plots/common.py): addiitional utils for plotting module
  - [plots - config](./wswan/plots/config.py): plotting configuration parameters
  - [plots - nonstationary](./wswan/plots/nonstationary.py): plotting toolbox for non-stationary cases and storm tracks / vortex model winds. 
  - [plots - stationary](./wswan/plots/stationary.py): plotting toolbox for stationary cases
  - [plots - stopmotion](./wswan/plots/stopmotion.py): plotting toolbox for stopmotion segments

## Documentation


## Jupyter Book

https://geoocean.gitlab.io/bluemath/numerical-models-wrappers/wrap_swan/book/intro.html

## Install
- - -

The source code is currently hosted on GitLab at: https://gitlab.com/geoocean/bluemath/numerical-models-wrappers/wrap_swan

This toolbox has been developed with Python 3.7.10

### Install from sources

Install requirements. Navigate to the base root of [wrap\_swan](./) and execute:

```bash
  pip install -r requirements.txt
```

Then install wrap\_swan:

```bash
  python setup.py install
```

### Install SWAN numerical model 

Download and Compile SWAN numerical model:

```bash
  # you may need to install a fortran compiler
  sudo apt install gfortran

  # download and unpack
  wget http://swanmodel.sourceforge.net/download/zip/swan4131.tar.gz
  tar -zxvf swan4131.tar.gz

  # compile numerical model
  cd swan4131/
  make config
  make ser
```

Copy SWAN binary file to module resources

```bash
  # Launch a python interpreter
  $ python

  Python 3.7.10 (default, Feb 27 2021, 02:19:57) 
  [Clang 12.0.0 (clang-1200.0.32.29)] on darwin
  Type "help", "copyright", "credits" or "license" for more information.
  
  >>> import wswan 
  >>> wswan.set_swan_binary_file('swan.exe')
```

## Examples:
- - -

- [demo 01 - stationary](./scripts/demo_01_stat.py): stationary example
- [demo 02 - non-stationary](./scripts/demo_02_nonstat.py): non-stationary example with input boundary waves, uniform winds and levels
- [demo 03 - Vortex model from parameters](./scripts/demo_03_nonstat_vortex_params.py): non-stationary with TCs Vortex Model example (from parameters)
- [demo 04 - Vortex model from historical track](./scripts/demo_04_nonstat_vortex_hist.py): non-stationary with TCs Vortex Model example (historical track interpolation)
- [demo 05 - non-stationary with uniform winds (cartesian)](./scripts/demo_05_nonstat_wind_xy.py): non-stationary with uniform winds (cartesian coordinates) 
- [demo 06 - non-stationary with uniform winds (spherical)](./scripts/demo_06_nonstat_wind_lonlat.py): non-stationary with uniform winds (spherical coordinates) 
- [demo 07 - non-stationary with customized 2D wind maps](./scripts/demo_07_nonstat_wind_2d.py): non-stationary with 2D wind maps as input 
- [demo 08 - non-stationary with waves at boundary segments](./scripts/demo_07_nonstat_wind_2d.py): non-stationary with input waves at user defined boundary segments 
- [notebook - SWAN Vortex Historical](./notebooks/nb_nonstat_vortex.ipynb): non-stationary storm Vortex model simulation from historical storm track

## Contributors:

Nicolás Ripoll Cabarga (nicolas.ripoll@unican.es)\
Alba Ricondo Cueva (ricondoa@unican.es)\
Sara Ortega Van Vloten (sara.ortegav@unican.es)\
Fernando Mendez Incera (fernando.mendez@unican.es)

## Thanks also to:

GeoOcean, FLTQ - University of Cantabria

## License

This project is licensed under the MIT License - see the [license](./LICENSE.txt) file for details

