# Introduction

SWAN numerical model python wrapper

WSWAN toolbox can be used to build and launch SWAN numerical model cases from a Python environment, and to extract model output once execution ends. 

Stationary cases are built from input waves and wind static conditions.

Non-stationary cases require a time series of wave conditions. Additional methodologies are included to build cases from storm tracks, using vortex numerical model for wind fields generation.

An alternative methodology for solving non-stationary wind cases by splitting storm tracks into segments is included in the stopmotion submodule. 

A plotting toolbox is included for case input and output visualization.

