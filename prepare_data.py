#Simple Multilayer Neural Network to seperate pycbc injections from noise triggers
#Author: Hunter Gabbard
#Max Planck Insitute for Gravitational Physics
#How to use only one GPU device....export CUDA_VISIBLE_DEVICES="0" in command line prior to run
#How to run in the background from the command line...python simple_neural_network.py -d NSBH01_ifar0-1.hdf,NSBH02_ifar0-1.hdf >/dev/null 2>err.txt &

from __future__ import division
import argparse
import numpy as np
import h5py
import sys
import os
from math import exp, log
import datetime


#Use seed value of 32 for testing purposes
np.random.seed(seed = 32)

#Get current time for time stamp labels
now = datetime.datetime.now()

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-o", "--output_dir", required=True,
        help="path to output directory")
args = vars(ap.parse_args())

data_files = args['dataset'].split(',')
out_dir = args['output_dir']

#Load in dataset parameters into variables and then combine into a numpy array
trig_comb = []
np.asarray(trig_comb)
h1 = h5py.File(args['dataset'].split(',')[0], 'r')
marg_l = np.asarray(h1['H1/marg_l'][:]).reshape((h1['H1/marg_l'].shape[0],1))
count = np.asarray(h1['H1/count'][:]).reshape((h1['H1/count'].shape[0],1))
maxnewsnr = np.asarray(h1['H1/maxnewsnr'][:]).reshape((h1['H1/maxnewsnr'].shape[0],1))
maxsnr = np.asarray(h1['H1/maxsnr'][:]).reshape((h1['H1/maxsnr'].shape[0],1))
time = np.asarray(h1['H1/time'][:]).reshape((h1['H1/time'].shape[0],1))
ratio_chirp = np.asarray(h1['H1/ratio_chirp'][:]).reshape((h1['H1/ratio_chirp'].shape[0],1))
delT = np.asarray(h1['H1/delT'][:]).reshape((h1['H1/delT'].shape[0],1))
delta_chirp = np.asarray(h1['H1/delta_chirp'][:]).reshape((h1['H1/delta_chirp'].shape[0],1))

trig_comb = np.hstack((marg_l, count, maxnewsnr, maxsnr, ratio_chirp, delT))

#load CBC/noise triggers and identify labels
for fi in data_files:
    h1 = h5py.File(fi, 'r')
    if args['dataset'].split(',')[0] == fi:
        marg_l_inj = np.asarray(h1['H1/marg_l_inj'][:]).reshape((h1['H1/marg_l_inj'].shape[0],1))
        count_inj = np.asarray(h1['H1/count_inj'][:]).reshape((h1['H1/count_inj'].shape[0],1))
        maxnewsnr_inj = np.asarray(h1['H1/maxnewsnr_inj'][:]).reshape((h1['H1/maxnewsnr_inj'].shape[0],1))
        maxsnr_inj = np.asarray(h1['H1/maxsnr_inj'][:]).reshape((h1['H1/maxsnr_inj'].shape[0],1))
        time_inj = np.asarray(h1['H1/time_inj'][:]).reshape((h1['H1/time_inj'].shape[0],1))
        ratio_chirp_inj = np.asarray(h1['H1/ratio_chirp_inj'][:]).reshape((h1['H1/ratio_chirp_inj'].shape[0],1))
        delT_inj = np.asarray(h1['H1/delT_inj'][:]).reshape((h1['H1/delT_inj'].shape[0],1))
        delta_chirp_inj = np.asarray(h1['H1/delta_chirp_inj'][:]).reshape((h1['H1/delta_chirp_inj'].shape[0],1))
        eff_dist_inj = np.asarray(h1['H1/eff_dist_inj'][:]).reshape((h1['H1/eff_dist_inj'].shape[0],1))

    else:
        marg_l_inj_new = np.asarray(h1['H1/marg_l_inj'][:]).reshape((h1['H1/marg_l_inj'].shape[0],1))
        marg_l_inj = np.vstack((marg_l_inj, marg_l_inj_new))
        count_inj_new = np.asarray(h1['H1/count_inj'][:]).reshape((h1['H1/count_inj'].shape[0],1))
        count_inj = np.vstack((count_inj, count_inj_new))
        maxnewsnr_inj_new = np.asarray(h1['H1/maxnewsnr_inj'][:]).reshape((h1['H1/maxnewsnr_inj'].shape[0],1))
        maxnewsnr_inj = np.vstack((maxnewsnr_inj, maxnewsnr_inj_new))
        maxsnr_inj_new = np.asarray(h1['H1/maxsnr_inj'][:]).reshape((h1['H1/maxsnr_inj'].shape[0],1))
        maxsnr_inj = np.vstack((maxsnr_inj, maxsnr_inj_new))
        time_inj_new = np.asarray(h1['H1/time_inj'][:]).reshape((h1['H1/time_inj'].shape[0],1))
        time_inj = np.vstack((time_inj, time_inj_new))
        ratio_chirp_inj_new = np.asarray(h1['H1/ratio_chirp_inj'][:]).reshape((h1['H1/ratio_chirp_inj'].shape[0],1))
        ratio_chirp_inj = np.vstack((ratio_chirp_inj, ratio_chirp_inj_new))
        delT_inj_new = np.asarray(h1['H1/delT_inj'][:]).reshape((h1['H1/delT_inj'].shape[0],1))
        delT_inj = np.vstack((delT_inj, delT_inj_new))
        delta_chirp_inj_new = np.asarray(h1['H1/delta_chirp_inj'][:]).reshape((h1['H1/delta_chirp_inj'].shape[0],1))
        delta_chirp_inj = np.vstack((delta_chirp_inj, delta_chirp_inj_new))
        eff_dist_inj_new = np.asarray(h1['H1/eff_dist_inj'][:]).reshape((h1['H1/eff_dist_inj'].shape[0],1))
        eff_dist_inj = np.vstack((eff_dist_inj, eff_dist_inj_new))


inj_weights_pre = []
np.asarray(inj_weights_pre)
dist_inj_mean = (eff_dist_inj**2).mean()
for idx in enumerate(delta_chirp_inj):
    idx = idx[0]
    inj_weights_pre.append((eff_dist_inj[idx][0]**2)/dist_inj_mean)

inj_weights = np.asarray(inj_weights_pre).reshape((delta_chirp_inj.shape[0],1))

#Retaining pre-normalized feature values for plots
inj_comb = np.hstack((marg_l_inj, count_inj, maxnewsnr_inj, maxsnr_inj, ratio_chirp_inj, delT_inj))
comb_all = np.vstack((trig_comb, inj_comb))
indices_trig = np.random.permutation(trig_comb.shape[0])
trig_train_idx, trig_test_idx = indices_trig[:int(trig_comb.shape[0]*.7)], indices_trig[int(trig_comb.shape[0]*.7):int(trig_comb.shape[0])]
trig_train_p, trig_test_p = trig_comb[trig_train_idx,:], trig_comb[trig_test_idx,:]
indices_inj = np.random.permutation(inj_comb.shape[0])
inj_train_idx, inj_test_idx = indices_inj[:int(inj_comb.shape[0]*.7)], indices_inj[int(inj_comb.shape[0]*.7):]
inj_train_p, inj_test_p = inj_comb[inj_train_idx,:], inj_comb[inj_test_idx,:]
train_data_p = np.vstack((trig_train_p, inj_train_p))
test_data_p = np.vstack((trig_test_p, inj_test_p))


#Normalizing
marg_l = ((comb_all[:,0] - comb_all[:,0].mean())/comb_all[:,0].max()).reshape((comb_all.shape[0],1))
count = ((comb_all[:,1] - comb_all[:,1].mean())/comb_all[:,1].max()).reshape((comb_all.shape[0],1))
maxnewsnr = ((comb_all[:,2] - comb_all[:,2].mean())/comb_all[:,2].max()).reshape((comb_all.shape[0],1))
maxsnr = ((comb_all[:,3] - comb_all[:,3].mean())/comb_all[:,3].max()).reshape((comb_all.shape[0],1))
ratio_chirp = ((comb_all[:,4] - comb_all[:,4].mean())/comb_all[:,4].max()).reshape((comb_all.shape[0],1))
delT = ((comb_all[:,5] - comb_all[:,5].mean())/comb_all[:,5].max()).reshape((comb_all.shape[0],1))
trig_comb = np.hstack((marg_l[0:trig_comb.shape[0]],count[0:trig_comb.shape[0]],maxnewsnr[0:trig_comb.shape[0]],maxsnr[0:trig_comb.shape[0]],ratio_chirp[0:trig_comb.shape[0]],delT[0:trig_comb.shape[0]]))
inj_comb = np.hstack((marg_l[trig_comb.shape[0]:],count[trig_comb.shape[0]:],maxnewsnr[trig_comb.shape[0]:],maxsnr[trig_comb.shape[0]:],ratio_chirp[trig_comb.shape[0]:],delT[trig_comb.shape[0]:]))
comb_all = np.vstack((trig_comb, inj_comb))

#Randomizing the order of the background triggers
indices_trig = np.random.permutation(trig_comb.shape[0])

#Seperate data into training and testing
trig_train_idx, trig_test_idx = indices_trig[:int(trig_comb.shape[0]*.7)], indices_trig[int(trig_comb.shape[0]*.7):int(trig_comb.shape[0])]
trig_train, trig_test = trig_comb[trig_train_idx,:], trig_comb[trig_test_idx,:]
indices_inj = np.random.permutation(inj_comb.shape[0])
inj_train_idx, inj_test_idx = indices_inj[:int(inj_comb.shape[0]*.7)], indices_inj[int(inj_comb.shape[0]*.7):]
inj_train_weight, inj_test_weight = inj_weights[inj_train_idx,:], inj_weights[inj_test_idx,:]
inj_train, inj_test = inj_comb[inj_train_idx,:], inj_comb[inj_test_idx,:]
train_data = np.vstack((trig_train, inj_train))
test_data = np.vstack((trig_test, inj_test))
train_data = train_data
test_data = test_data

#making labels (zero is noise, one is injection)
c_zero = np.zeros((trig_comb.shape[0],1))
c_z_train = c_zero[:int(trig_comb.shape[0]*.7)]
c_z_test = c_zero[int(trig_comb.shape[0]*.7):int(trig_comb.shape[0])]
c_ones = np.ones((int(inj_comb.shape[0]),1))
c_o_train = c_ones[:int(inj_comb.shape[0]*.7)]
c_o_test = c_ones[int(inj_comb.shape[0]*.7):int(inj_comb.shape[0])]
lab_train = np.vstack((c_z_train,c_o_train))
lab_test = np.vstack((c_z_test,c_o_test))
labels_all = np.vstack((c_zero,c_ones))

#Creating sample weights vector
trig_weights = np.zeros((trig_comb.shape[0],1))
trig_weights.fill(1/((trig_comb.shape[0])/(inj_comb.shape[0])))
trig_w_train = trig_weights[:int(trig_comb.shape[0]*.7)]
trig_w_test = trig_weights[int(trig_comb.shape[0]*.7):]
train_weights = np.vstack((trig_w_train,inj_train_weight)).flatten()

#Write data to an hdf file
with h5py.File('nn_data.hdf', 'w') as hf:
    hf.create_dataset('test_data', data=test_data)
    hf.create_dataset('train_data', data=train_data)
    hf.create_dataset('train_weights', data=train_weights)
    hf.create_dataset('lab_test', data=lab_test)
    hf.create_dataset('lab_train', data=lab_train)


