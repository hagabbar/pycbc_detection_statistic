import argparse
import h5py
import sys, os
import numpy as np
import unicodedata

parser = argparse.ArgumentParser()
parser.add_argument('--orig-files', nargs='+')
parser.add_argument('--new-files', nargs='+')
parser.add_argument('--ifo', type=str)

args = parser.parse_args()
orig_files = args.orig_files
new_files = args.new_files
ifo = args.ifo
os.mkdir('../data/O1/%s/chunk2/new_files' % ifo)

for idx,fi in enumerate(orig_files):
    f1 = h5py.File(fi, 'r')
    f2 = h5py.File(new_files[idx], 'r')
    opt_snr = f2['%s/opt_snr' % ifo][:] 
    
    #Write data to an hdf file
    up_fi = h5py.File('../data/O1/%s/chunk2/new_files/%s' %( ifo,fi.split('/')[5]), 'w')       
    up_fi['%s/opt_snr' % ifo] = opt_snr
    for idx2,key in enumerate(f1['%s' % ifo].keys()):
        key = f1['L1'].keys()[idx2]
        old_key = f1['%s/%s' % (ifo,key)][:]
        
        up_fi['%s/%s' % (ifo,key)] = old_key

