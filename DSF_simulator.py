import os
import sys
import math
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from phonopy import load

def generate_qpoints(n_sample, r_max, prim_latt):
    rand_array = np.random.rand(3, n_sample)
    sphere_coords = np.zeros((3, n_sample))
    sphere_coords[0] = r_max * np.cbrt(rand_array[0]) # r
    sphere_coords[1] = 2 * np.pi * rand_array[1]  # theta
    sphere_coords[2] = np.arccos(2*rand_array[2]-1) # phi

    cartesian_coords = np.zeros((3, n_sample))
    cartesian_coords[0] = sphere_coords[0] * np.sin(sphere_coords[2]) * np.cos(sphere_coords[1]) # x
    cartesian_coords[1] = sphere_coords[0] * np.sin(sphere_coords[2]) * np.sin(sphere_coords[1]) # y
    cartesian_coords[2] = sphere_coords[0] * np.cos(sphere_coords[2]) # z

    recip_coords = np.zeros((4, n_sample))
    recip_coords[0] = sphere_coords[0] # keep r value
    recip_coords[1:] = np.dot(prim_latt, cartesian_coords) / (2*np.pi) # primitive reciprocal

    return recip_coords


def compute_dsf(phonon, qpoints, temperature, scattering_lengths):
    # for INS, scattering length has to be given.
    # the following values is obtained at (Coh b)
    # https://www.nist.gov/ncnr/neutron-scattering-lengths-list
    phonon.run_dynamic_structure_factor(
        qpoints[1:].T,
        temperature,
        scattering_lengths=scattering_lengths,
        freq_min=1e-3)
    dsf = phonon.dynamic_structure_factor

    # collect output
    data_out = []
    for ipoint in range(qpoints.shape[1]):
        Q = qpoints[0, ipoint]
        frequencies = dsf.frequencies[ipoint]
        dsfs = dsf.dynamic_structure_factors[ipoint]
        for ifreq in range(len(frequencies)):
            data_out.append([Q, frequencies[ifreq], dsfs[ifreq]])
    
    return data_out
    

if __name__ == '__main__':
    # output file
    rand_ID = int(np.random.rand() * 10**8)
    out_file = "ID-"+str(rand_ID)+".npy"

    # read phonon data, requires FORCE_SETS in directory
    phonon = load(supercell_filename="SPOSCAR")
    prim_latt = phonon.primitive.get_cell()
    # mesh sampling phonon calculation is needed for Debye-Waller factor
    # this must be done with is_mesh_symmetry=False and with_eigenvectors=True
    mesh = [11, 11, 11]
    phonon.run_mesh(mesh, is_mesh_symmetry=False,
                          with_eigenvectors=True)

    # set parameters
    n_sample = 1000 # number of uniform sampling k-points in sphere
    r_max = 9 # maximum |Q|
    temperature = 5 # for Debye-Waller factor
    scattering_lengths = {'Ga': 7.288, 'Ta': 6.91, 'Se': 7.97}
    print("number of sampling points:", n_sample, flush=True)

    # sampling q-points
    qpoints = generate_qpoints(n_sample, r_max, prim_latt)

    # simulate DSF
    start = time.time()
    dsf_data = compute_dsf(phonon, qpoints, temperature, scattering_lengths)
    end = time.time()
    print('Time elapsed on DSF per qpoint: {}s'.format((end-start)/n_sample))
 
    data_out = np.array(dsf_data)
    np.save(out_file, data_out)

    print("Program finished normally")


