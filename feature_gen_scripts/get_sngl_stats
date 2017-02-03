#!/usr/bin/env python
#Script to generate features for pycbc neural network detection statist
#Author: Hunter Gabbard and Tom Dent
#Max Planck Institute for Gravitational Physics
#Example command line run: python get_sngl_stats --ifo H1 --single-trigger-files H1-HDF_TRIGGER_MERGE_FULL_DATA-1128299417-1083600.hdf --veto-file H1L1-CUMULATIVE_CAT_12H_VETO_SEGMENTS.xml --veto-segment-name CUMULATIVE_CAT_12H --found-injection-file H1L1-HDFINJFIND_BBH01_INJ_INJ_INJ-1128299417-1083600.hdf --window 1 --output-file BBH01_test.hdf --temp-bank H1L1-BANK2HDF-1128299417-1083600.hdf --inj-file H1-HDF_TRIGGER_MERGE_BBH01_INJ-1128299417-1083600.hdf --inj-coinc-file H1L1-HDFINJFIND_BBH01_INJ_INJ_INJ-1128299417-1083600.hdf --ifar-thresh 0.1 --verbose

import argparse
import numpy
import h5py
import logging
import pycbc
from scipy.misc import logsumexp
from pycbc import events, init_logging, pnutils
import sympy
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from math import log
from scipy.misc import logsumexp
import pycbc
from pycbc import events
import sympy
from pycbc.types import MultiDetOptionAction
import matplotlib.pyplot as plt


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
         
        time_injc = time_injc[ifar_injc > float(args.ifar_thresh)]
        injection_index_injc = hf_injcoinc['found_after_vetoes/injection_index'][ifar_injc > float(args.ifar_thresh)]
        dist_injc = dist_injc[injection_index_injc] 
        
	del ifar_injc

        time = f['end_time'][:]
        snr = f['snr'][:]
        template_dur = f['template_duration'][:]
	template_id = f['template_id'][:]
        chisq = f['chisq'][:]
        chisq_dof = f['chisq_dof'][:]
        rchisq = chisq / (2 * chisq_dof - 2)
        del chisq
        del chisq_dof
	newsnr = events.newsnr(snr, rchisq)
	del rchisq

        mass1_full = mass1[template_id]
        mass2_full = mass2[template_id]

	f_inj = hf_inj[ifo]
	time_inj = f_inj['end_time'][:]
	snr_inj = f_inj['snr'][:]
        template_dur_inj = f_inj['template_duration'][:]
        template_id_inj = f_inj['template_id'][:]
	chisq_inj = f_inj['chisq'][:]
        chisq_dof_inj = f_inj['chisq_dof'][:]
        rchisq_inj = chisq_inj / (2 * chisq_dof_inj - 2)
        del chisq_inj
        del chisq_dof_inj
        newsnr_inj = events.newsnr(snr_inj, rchisq_inj)
        del rchisq_inj

        mass1_inj = mass1[template_id_inj]
        mass2_inj = mass2[template_id_inj]

        # apply vetoes
        try: 
            len(args.veto_file)
        except TypeError:
            print "NO VETO FILE"
	    #After retrieving clean set, iterate and sort through
            tsort = time.argsort()
            time = time[tsort]
            snr = snr[tsort]
            template_dur = template_dur[tsort]
	    template_id = template_id[tsort]
            newsnr = newsnr[tsort]

	    #After retrieving clean set, iterate and sort through
            tsort_inj = time_inj.argsort()
            time_inj = time_inj[tsort_inj]
            snr_inj = snr_inj[tsort_inj]
            template_dur_inj = template_dur_inj[tsort_inj]
            newsnr_inj = newsnr_inj[tsort_inj]
            template_id_inj = template_id_inj[tsort_inj]
        else:
            print "sure, it was defined"
            mask, segs = events.veto.indices_outside_segments(time, [args.veto_file], ifo=ifo,
                segment_name=args.veto_segment_name)
	
	    #After retrieving clean set, iterate and sort through triggers
            tsort = time[mask].argsort()
	    print len(tsort)
	    print len(time.argsort())
            time = time[mask][tsort]
            snr = snr[mask][tsort]
            template_dur = template_dur[mask][tsort]
	    newsnr = newsnr[mask][tsort]
            template_id = template_id[mask][tsort]

	    #For injections...
	    mask_inj, segs_inj = events.veto.indices_outside_segments(time_inj, [args.veto_file], ifo=ifo,
                segment_name=args.veto_segment_name)

            #After retrieving clean set, iterate and sort through triggers
            tsort_inj = time_inj[mask_inj].argsort()
            time_inj = time_inj[mask_inj][tsort_inj]
            snr_inj = snr_inj[mask_inj][tsort_inj]
            template_dur_inj = template_dur_inj[mask_inj][tsort_inj]
            newsnr_inj = newsnr_inj[mask_inj][tsort_inj]
            template_id_inj = template_id_inj[mask_inj][tsort_inj]

	#Take only triggers found to be in coincidence
        like = snr ** 2.0 / 2
        logging.info('%s triggers', len(time))

	like_inj = snr_inj ** 2.0 / 2

        # find the edges of the integral
        window_times = numpy.arange(int(time[0] / int(args.window)), int(time[-1] / int(args.window)) + 1) * int(args.window)
	left = numpy.searchsorted(time, window_times - int(args.window))
        right = numpy.searchsorted(time, window_times + int(args.window))

        left_inj = numpy.searchsorted(time_inj, window_times - int(args.window))
        right_inj = numpy.searchsorted(time_inj, window_times + int(args.window)) 

	
        margl = []
        count = []
        maxsnr = []
        maxnewsnr = []
        tmp_dur = []
        t = []
	delT = []
        delta_chirp = []
        ratio_chirp = []
        #eff_dist = []

	margl_inj = []
        count_inj = []
        maxsnr_inj = []
        maxnewsnr_inj = []
        tmp_dur_inj = []
	t_inj = []
	delT_inj = []
        delta_chirp_inj = []
        ratio_chirp_inj = []
        dist_inj = []
        chirp_m_inj = []

        for i, (l, r) in enumerate(zip(left, right)):
            logging.info('%s: %2.1f complete', ifo, float(i) / len(right) * 100)
            if r > l:
                
                t.append(window_times[i])
                margl.append(logsumexp(like[l:r]))
                count.append(r - l)
                maxsnr.append(snr[l:r].max())
		idx_mxsnr = numpy.argmax(snr[l:r])
                maxnewsnr.append(newsnr[l:r].max())
		idx_mxnewsnr = numpy.argmax(newsnr[l:r])
		deltaT_trig = time[l:r][idx_mxnewsnr] - time[l:r][idx_mxsnr]
                delT.append(deltaT_trig)
                tmp_dur.append(template_dur[l:r][idx_mxsnr])
                
                
                #determining the chirp mass
                m1_snr = mass1_full[l:r][idx_mxsnr] 
                m2_snr = mass2_full[l:r][idx_mxsnr]
                m1_new_snr = mass1_full[l:r][idx_mxnewsnr]                       
                m2_new_snr = mass2_full[l:r][idx_mxnewsnr]
                chirp_snr = pnutils.mass1_mass2_to_mchirp_eta(m1_snr, m2_snr)
                chirp_new_snr = pnutils.mass1_mass2_to_mchirp_eta(m1_new_snr, m2_new_snr)
               

                #Calculating delta chirp and ratio of chirp masses
                delta_chirp.append(log(chirp_snr[0]) - log(chirp_new_snr[0]))
                ratio_chirp.append(chirp_snr[0]/chirp_new_snr[0])
	
	for i, (l, r) in enumerate(zip(left_inj, right_inj)):
            logging.info('%s: %2.1f complete', ifo, float(i) / len(right_inj) * 100)
            closest_inj_id = numpy.argmin(abs(time_injc - window_times[i]))
            closest_injt = time_injc[closest_inj_id]

            #The inj_coinc id is given by closest_inj_id variable
            closest_injd = dist_injc[closest_inj_id]

            if r > l and closest_injt+0.5 >= window_times[i] and closest_injt-0.5 < window_times[i]:
                t_inj.append(window_times[i])
                margl_inj.append(logsumexp(like_inj[l:r]))
                count_inj.append(r - l)
                maxsnr_inj.append(snr_inj[l:r].max())
		idx_mxsnr = numpy.argmax(snr_inj[l:r])
                maxnewsnr_inj.append(newsnr_inj[l:r].max())
                idx_mxnewsnr = numpy.argmax(newsnr_inj[l:r])
		deltaT_inj = time_inj[l:r][idx_mxnewsnr] - time_inj[l:r][idx_mxsnr]
                delT_inj.append(deltaT_inj)
                tmp_dur_inj.append(template_dur_inj[l:r][idx_mxsnr])

		#determining the chirp mass
	        m1_snr = mass1_inj[l:r][idx_mxsnr]
                m2_snr = mass2_inj[l:r][idx_mxsnr]
                m1_new_snr = mass1_inj[l:r][idx_mxnewsnr]
                m2_new_snr = mass2_inj[l:r][idx_mxnewsnr]
                chirp_snr = pnutils.mass1_mass2_to_mchirp_eta(m1_snr, m2_snr)
                chirp_m_inj.append(chirp_snr)
                chirp_new_snr = pnutils.mass1_mass2_to_mchirp_eta(m1_new_snr, m2_new_snr)
                dist_inj.append(closest_injd)
                #dist_inj.append((chirp_new_snr[0]**(5/6))/(snr_inj[l:r].max()))
            

                #Calculating delta chirp and ratio of chirp masses
                delta_chirp_inj.append(log(chirp_snr[0]) - log(chirp_new_snr[0]))
                ratio_chirp_inj.append(chirp_snr[0]/chirp_new_snr[0])

        o['%s/chirp_m_inj' % ifo] = numpy.array(chirp_m_inj)
        o['%s/dist_inj' % ifo] = numpy.array(dist_inj)
	o['%s/delta_chirp' % ifo] = numpy.array(delta_chirp)
	o['%s/delta_chirp_inj' % ifo] = numpy.array(delta_chirp_inj)
        o['%s/delT' % ifo] = numpy.array(delT)
        o['%s/delT_inj' % ifo] = numpy.array(delT_inj)
        o['%s/ratio_chirp' % ifo] = numpy.array(ratio_chirp)
        o['%s/ratio_chirp_inj' % ifo] = numpy.array(ratio_chirp_inj)       
        o['%s/time' % ifo] = numpy.array(t)
        o['%s/marg_l' % ifo] = numpy.array(margl)
        o['%s/count' % ifo] = numpy.array(count)
        o['%s/maxsnr' % ifo] = numpy.array(maxsnr)
        o['%s/maxnewsnr' % ifo] = numpy.array(maxnewsnr)
	o['%s/time_inj' % ifo] = numpy.array(t_inj)
        o['%s/marg_l_inj' % ifo] = numpy.array(margl_inj)
        o['%s/count_inj' % ifo] = numpy.array(count_inj)
        o['%s/maxsnr_inj' % ifo] = numpy.array(maxsnr_inj)
        o['%s/maxnewsnr_inj' % ifo] = numpy.array(maxnewsnr_inj)
        o['%s/template_duration' % ifo] = numpy.array(tmp_dur)
        o['%s/template_duration_inj' % ifo] = numpy.array(tmp_dur_inj)



print "Forget the coincs, I'm outta here"
exit()

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
