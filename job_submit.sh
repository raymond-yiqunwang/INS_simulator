#!/bin/bash
#
# Required PBS Directives --------------------------------------
#
#PBS -A ARONC33993043
#PBS -q standard
#PBS -l select=1:ncpus=32:mpiprocs=32
#PBS -l walltime=01:00:00

## Optional PBS Directives --------------------------------------
#PBS -j oe
#PBS -V
#PBS -S /bin/bash
#PBS -N GNS

# Execution Block ----------------------------------------------
# Environment Setup
# cd to your scratch directory in /scr
cd ${PBS_O_WORKDIR}

## job specific

aprun -n 32 /u/raymondw/miniconda2/envs/phonopy/bin/python DSF_copper.py > job.log 

