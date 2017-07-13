#!/usr/bin/env python
#Script to generate features for pycbc neural network detection statist
#Author: Hunter Gabbard and Tom Dent
#Max Planck Institute for Gravitational Physics
#Example command line run: python get_sngl_stats --ifo H1 --single-trigger-files H1-HDF_TRIGGER_MERGE_FULL_DATA-1128299417-1083600.hdf --veto-file H1L1-CUMULATIVE_CAT_12H_VETO_SEGMENTS.xml --veto-segment-name CUMULATIVE_CAT_12H --found-injection-file H1L1-HDFINJFIND_BBH01_INJ_INJ_INJ-1128299417-1083600.hdf --window 1 --output-file BBH01_test.hdf --temp-bank H1L1-BANK2HDF-1128299417-1083600.hdf --inj-file H1-HDF_TRIGGER_MERGE_BBH01_INJ-1128299417-1083600.hdf --inj-coinc-file H1L1-HDFINJFIND_BBH01_INJ_INJ_INJ-1128299417-1083600.hdf --ifar-thresh 0.1 --verbose

import argparse
import sys
import numpy
import h5py
import logging
import pycbc
from scipy.misc import logsumexp
from pycbc import events, init_logging, pnutils
import sympy
#import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from math import log
from scipy.misc import logsumexp
import pycbc
from pycbc import events
import sympy
from pycbc.types import MultiDetOptionAction
#import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--single-trigger-files')
parser.add_argument('--ifo')
#parser.add_argument('--single-trigger-files', nargs='+')
parser.add_argument('--veto-file')
parser.add_argument('--veto-segment-name')
parser.add_argument('--found-injection-file')
#parser.add_argument('--coinc-triggers', nargs='+')
#parser.add_argument('--coinc-threshold', type=float)
parser.add_argument('--window', type=float)
parser.add_argument('--output-file')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--inj-file')
parser.add_argument('--inj-coinc-file')
parser.add_argument('--ifar-thresh')
parser.add_argument('--temp-bank')
parser.add_argument('--just-inj', type=str, default = 'False')

args = parser.parse_args()
pycbc.init_logging(args.verbose)
o = h5py.File(args.output_file, 'w')
snrs = {}

# NOTE : Use found injection file to restrict injection stats to windows
# surrounding 'well found' coincident injection triggers ..

# SINGLES ####################################################################
logging.info('Handling Single Detector Triggers')

with h5py.File(args.single_trigger_files, 'r') as hf, h5py.File(args.inj_file, 'r') as hf_inj, h5py.File(args.inj_coinc_file, 'r') as hf_injcoinc, h5py.File(args.temp_bank, 'r') as hf_t_bank:
    for ifo in hf:
        f = hf[ifo]
	f_injc = hf_injcoinc['found_after_vetoes']
	
	if ifo == 'H1':
            ifonum = '2'
        elif ifo == 'L1':
            ifonum = '1'

        #Downloading template bank masses
        mass1 = hf_t_bank['mass1'][:]
        mass2 = hf_t_bank['mass2'][:]

        #Change eff_dist to dist...will confuse people
	#Storing coinc_injs and removing those with < 0.1 ifar
	time_injc = f_injc['time'+ifonum][:]
	ifar_injc = f_injc['ifar'][:]
        dist_injc = hf_injcoinc['injections/distance'][:]

        #Add in optimal snr feature for use as a new weight prior to training neural network
        if ifo == 'L1':
            opt_snr = hf_injcoinc['injections/optimal_snr_1'][:]
        elif ifo == 'H1':
            opt_snr = hf_injcoinc['injections/optimal_snr_2'][:]

        injection_index_injc = hf_injcoinc['found_after_vetoes/injection_index'][ifar_injc > float(args.ifar_thresh)] 
        opt_snr_final = 1./(opt_snr[injection_index_injc]**2)
  

        o['%s/opt_snr' % ifo] = numpy.array(opt_snr_final)

 
print "Forget the coincs, I'm outta here"
sys.exit()

# COINCS ######################################################################
# Assumes that *all* the coincs above some threshold are available.
logging.info('Handling the Coincident Triggers')
logging.info ('Collecting coincs')

t1 = []
t2 = []
ts = []
tid = {'H1':[], 'L1':[]}
for fname in args.coinc_triggers:
    f = h5py.File(fname)
    logging.info('reading %s', fname)
    stat = f['stat'][:]
    l = stat > args.coinc_threshold
    t1.append(f['time1'][:][l])
    t2.append(f['time2'][:][l])
    ts.append(f['timeslide_id'][:][l])
    tid[f.attrs['detector_1']].append(f['trigger_id1'][:][l])
    tid[f.attrs['detector_2']].append(f['trigger_id2'][:][l])

t1 = numpy.concatenate(t1)
t2 = numpy.concatenate(t2)
ts = numpy.concatenate(ts)
slide = f.attrs['timeslide_interval']

for ifo in tid:
    tid[ifo] = numpy.concatenate(tid[ifo])

logging.info('Start Sorting')
time = (t2 + (t1 + ts * slide)) / 2

time = time.astype(numpy.float128)

span = int((time.max() - time.min()) / args.window) * float(args.window) + args.window * 10
time = time + span * ts.astype(numpy.float128)

time_sorting = time.argsort()
time = time[time_sorting]
ts = ts[time_sorting]
for ifo in tid:
    tid[ifo] = tid[ifo][time_sorting]

window_pos = (time / args.window).astype(numpy.int64)
window_pos = numpy.unique(numpy.concatenate([window_pos, window_pos + 1]))
window_pos = window_pos.astype(numpy.float128) * args.window

left = numpy.searchsorted(time, window_pos - args.window)
right = numpy.searchsorted(time, window_pos + args.window)

k = (right - left) > 0
left = left[k]
right = right[k]
window_pos = window_pos[k]

print time[left].max(), time[left].min()
print time[right-1].max(), time[right-1].min()
print time.max(), time.min()

vs = []
ws = []

logging.info('Done Sorting')
for i, (l, r) in enumerate(zip(left, right)):
    htid = tid['H1'][l:r]
    ltid = tid['L1'][l:r]
    ws.append(i)
    fac = snrs['H1'][htid]**2.0 / 2.0 + snrs['L1'][ltid]**2.0 / 2.0
    go = mstat(fac)
    vs.append(go)
    logging.info('COINC: %2.2f', float(i) / len(right) * 100)
vs = numpy.array(vs)
ws = numpy.array(ws)

print window_pos.max(), window_pos.min()
print window_pos[ws].max(), window_pos[ws].min()
print time.max()

o['coinc/value'] = vs
o['coinc/time'] = (window_pos[ws] / args.window).astype(numpy.int64)
o['coinc'].attrs['span'] = span
o['coinc'].attrs['window'] = args.window
