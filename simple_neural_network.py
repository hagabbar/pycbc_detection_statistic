#Simple Multilayer Neural Network to seperate pycbc injections from noise triggers
#Author: Hunter Gabbard
#Max Planck Insitute for Gravitational Physics
#How to use only one GPU device....export CUDA_VISIBLE_DEVICES="0" in command line prior to run
#How to run in the background from the command line...python simple_neural_network.py -d NSBH01_ifar0-1.hdf,NSBH02_ifar0-1.hdf >/dev/null 2>err.txt &

from __future__ import division
import argparse
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, ActivityRegularization
from keras.optimizers import SGD
import sys
import os
from math import exp, log
import tensorflow as tf
from keras.callbacks import EarlyStopping
from matplotlib import use
use('Agg')
from matplotlib import pyplot as pl
import datetime


#Configure tensorflow to use gpu memory as needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
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


#Definition for loading in dataset parameters into variables and then combine into a numpy array
def load_back_data(data, params):
    trig_comb = {}
    params = ['marg_l','count','maxnewsnr','maxsnr','time','ratio_chirp','delT','delta_chirp','template_duration']
    h1 = h5py.File(data.split(',')[0], 'r')
    for label in h1['H1'].keys():
        for key in params:
            if label == key:
                trig_comb[label] = np.asarray(h1['H1/%s' % label][:]).reshape((h1['H1/%s' % label].shape[0],1))
    return trig_comb

#Load CBC/noise triggers from multiple data sets
def load_inj_data(data, params):
    trig_comb = {}
    params = ['marg_l','count','maxnewsnr','maxsnr','time','ratio_chirp','delT','delta_chirp','template_duration']
    for fi in data:
        h1 = h5py.File(fi, 'r')
        if data.split(',')[0] == fi:
            for label in h1['H1'].keys():
                for key in params:
                    if label == key:
                        trig_comb[label] = np.asarray(h1['H1/%s' % label][:]).reshape((h1['H1/%s' % label].shape[0],1))
        else:
            for label in h1['H1'].keys():
                for key in params:
                    if label == key:
                        trig_comb[label+'_inj_new'] = np.asarray(h1['H1/%s' % label][:]).reshape((h1['H1/%s' % label].shape[0],1)) 
                        trig_comb[label+'_inj'] = np.vstack((trig_comb[label+'_inj'], trig_comb[label+'_inj_new']))

    return trig_comb

#Load in dataset parameters into variables and then combine into a numpy array
#converted
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
tmp_dur = np.asarray(h1['H1/template_duration'][:]).reshape((h1['H1/template_duration'].shape[0],1))

trig_comb = np.hstack((marg_l, count, maxnewsnr, maxsnr, ratio_chirp, delT, tmp_dur))

#load CBC/noise triggers and identify labels
#converted
for fi in data_files:
    print fi
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
        dist_inj = np.asarray(h1['H1/dist_inj'][:]).reshape((h1['H1/dist_inj'].shape[0],1))
        tmp_dur_inj = np.asarray(h1['H1/template_duration_inj'][:]).reshape((h1['H1/template_duration_inj'].shape[0],1))

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
        dist_inj_new = np.asarray(h1['H1/dist_inj'][:]).reshape((h1['H1/dist_inj'].shape[0],1))
        dist_inj = np.vstack((dist_inj, dist_inj_new))
        tmp_dur_inj_new = np.asarray(h1['H1/template_duration_inj'][:]).reshape((h1['H1/template_duration_inj'].shape[0],1))
        tmp_dur_inj = np.vstack((tmp_dur_inj, tmp_dur_inj_new))

#Getting injection weights
inj_weights_pre = []
np.asarray(inj_weights_pre)
dist_inj_mean = (dist_inj**2).mean()
for idx in enumerate(delta_chirp_inj):
    idx = idx[0]
    inj_weights_pre.append((dist_inj[idx][0]**2)/dist_inj_mean)

inj_weights = np.asarray(inj_weights_pre).reshape((delta_chirp_inj.shape[0],1))

#Retaining pre-normalized feature values for plots
def orig_norm(back_trig, inj_trig, tt_split):
    inj_trig = np.array(inj_trig.items(), dtype=dtype) 
    comb_all = np.vstack((back_trig, inj_trig))
    indices_trig = np.random.permutation(back_trig.shape[0])
    trig_train_idx, trig_test_idx = indices_trig[:int(back_trig.shape[0]*tt_split)], indices_trig[int(back_trig.shape[0]*tt_split):int(back_trig.shape[0])]
    trig_train_p, trig_test_p = back_trig[trig_train_idx,:], back_trig[trig_test_idx,:]
    indices_inj = np.random.permutation(inj_trig.shape[0])
    inj_train_idx, inj_test_idx = indices_inj[:int(inj_trig.shape[0]*tt_split)], indices_inj[int(inj_trig.shape[0]*tt_split):]
    inj_train_p, inj_test_p = inj_trig[inj_train_idx,:], inj_trig[inj_test_idx,:]
    train_data_p = np.vstack((trig_train_p, inj_train_p))
    test_data_p = np.vstack((trig_test_p, inj_test_p))
    return train_data_p, test_data_p

#Retaining pre-normalized feature values for plots
#converted
inj_comb = np.hstack((marg_l_inj, count_inj, maxnewsnr_inj, maxsnr_inj, ratio_chirp_inj, delT_inj, tmp_dur_inj))
comb_all = np.vstack((trig_comb, inj_comb))
indices_trig = np.random.permutation(trig_comb.shape[0])
trig_train_idx, trig_test_idx = indices_trig[:int(trig_comb.shape[0]*.7)], indices_trig[int(trig_comb.shape[0]*.7):int(trig_comb.shape[0])]
trig_train_p, trig_test_p = trig_comb[trig_train_idx,:], trig_comb[trig_test_idx,:]
indices_inj = np.random.permutation(inj_comb.shape[0])
inj_train_idx, inj_test_idx = indices_inj[:int(inj_comb.shape[0]*.7)], indices_inj[int(inj_comb.shape[0]*.7):]
inj_train_p, inj_test_p = inj_comb[inj_train_idx,:], inj_comb[inj_test_idx,:]
train_data_p = np.vstack((trig_train_p, inj_train_p))
test_data_p = np.vstack((trig_test_p, inj_test_p))

#def normalize():
#Normalizing
marg_l = ((np.log(comb_all[:,0]) - np.log(comb_all[:,0]).mean())/np.log(comb_all[:,0]).max()).reshape((comb_all.shape[0],1))
count = ((comb_all[:,1] - comb_all[:,1].mean())/comb_all[:,1].max()).reshape((comb_all.shape[0],1))
maxnewsnr = ((np.log(comb_all[:,2]) - np.log(comb_all[:,2]).mean())/np.log(comb_all[:,2]).max()).reshape((comb_all.shape[0],1))
maxsnr = ((np.log(comb_all[:,3]) - np.log(comb_all[:,3]).mean())/np.log(comb_all[:,3]).max()).reshape((comb_all.shape[0],1))
ratio_chirp = ((np.log(comb_all[:,4]) - np.log(comb_all[:,4]).mean())/np.log(comb_all[:,4]).max()).reshape((comb_all.shape[0],1))
delT = ((comb_all[:,5] - comb_all[:,5].mean())/comb_all[:,5].max()).reshape((comb_all.shape[0],1))
tmp_dur = ((np.log(comb_all[:,6]) - np.log(comb_all[:,6]).mean())/np.log(comb_all[:,6]).max()).reshape((comb_all.shape[0],1))
trig_comb = np.hstack((marg_l[0:trig_comb.shape[0]],count[0:trig_comb.shape[0]],maxnewsnr[0:trig_comb.shape[0]],maxsnr[0:trig_comb.shape[0]],ratio_chirp[0:trig_comb.shape[0]],delT[0:trig_comb.shape[0]],tmp_dur[0:trig_comb.shape[0]]))
inj_comb = np.hstack((marg_l[trig_comb.shape[0]:],count[trig_comb.shape[0]:],maxnewsnr[trig_comb.shape[0]:],maxsnr[trig_comb.shape[0]:],ratio_chirp[trig_comb.shape[0]:],delT[trig_comb.shape[0]:],tmp_dur[trig_comb.shape[0]:]))
comb_all = np.vstack((trig_comb, inj_comb))

#Randomizing the order of the background triggers
indices_trig = np.random.permutation(trig_comb.shape[0])

#Separate data into training and testing
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

# define the architecture of the network (sigmoid nodes)
model = Sequential()
act = keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.add(Dense(200, input_dim=trig_comb.shape[1]))
act

#model.add(Dense(25, init='normal', input_dim=6, activation='relu'))
#model.add(Dense(21, init='normal', activation='relu'))
#model.add(Dense(21, init='normal', activation='relu'))

#model.add(Dense(3, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(3, activation='relu'))


model.add(Dense(300))
act
#model.add(Dropout(0.2))
model.add(Dense(500))
act
#model.add(Dropout(0.2))
model.add(Dense(700))
act
#model.add(Dropout(0.2))
model.add(Dense(500))
act
#model.add(Dropout(0.2))
model.add(Dense(200))
act

model.add(Dense(1, init='normal', activation='sigmoid'))

#Compiling model
print("[INFO] compiling model...")
sgd = SGD(lr=0.05)
model.compile(loss="binary_crossentropy", optimizer='rmsprop',
	metrics=["accuracy","binary_crossentropy"], class_mode='binary')

#Creating sample weights vector
trig_weights = np.zeros((trig_comb.shape[0],1))
trig_weights.fill(1/((trig_comb.shape[0])/(inj_comb.shape[0])))
trig_w_train = trig_weights[:int(trig_comb.shape[0]*.7)]
trig_w_test = trig_weights[int(trig_comb.shape[0]*.7):]
train_weights = 100.*np.vstack((trig_w_train,inj_train_weight)).flatten()
test_weights = 100.*np.vstack((trig_w_test,inj_test_weight)).flatten()

#model.fit(train_data, lab_train, nb_epoch=1, batch_size=32, sample_weight=train_weights, shuffle=True, show_accuracy=True)
hist = model.fit(train_data, lab_train,
                    nb_epoch=1000, batch_size=100000,    #66000
                    sample_weight=train_weights,
                    validation_data=(test_data,lab_test,test_weights),
                    shuffle=True, show_accuracy=True)
print(hist.history)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
eval_results = model.evaluate(test_data, lab_test,
                                    sample_weight=test_weights,
	                                batch_size=32, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(eval_results[0],
	eval_results[1] * 100))
#Saving prediction probabilities to a variable
res_pre = model.predict(test_data)

#Printing summary of model parameters
model.summary()


#####################
#Computing ROC Curve#
#####################

#Sort trig and inj test features into same order as NN score vector will be
n_noise = len(trig_test)
pred_prob = model.predict_proba(test_data, batch_size=32).T[0]
prob_sort_noise = pred_prob[pred_prob[0:n_noise].argsort()][::-1]
prob_sort_inj = pred_prob[n_noise:][pred_prob[n_noise:].argsort()][::-1]
prob_sort_injWeight = inj_test_weight.T[0][pred_prob[n_noise:].argsort()][::-1]
prob_sort_injNewsnr = inj_test[:,2][pred_prob[n_noise:].argsort()][::-1]


newsnr_sort_noiseNewsnr = trig_test[:,2][trig_test[:,2][0:].argsort()][::-1]
newsnr_sort_injNewsnr = inj_test[:,2][inj_test[:,2][0:].argsort()][::-1]
newsnr_sort_injWeight = inj_test_weight.T[0][inj_test[:,2][0:].argsort()][::-1]

pred_class = model.predict_classes(test_data)
class_sort = pred_class[pred_prob[:].argsort()][::-1]

orig_test_labels = lab_test[pred_prob[:].argsort()][::-1]


#Function to calculate ROC values
def ROC(inj_weight, inj_param, noise_param):

    #Assert that length of inj weights and injection parameters are the same
    assert len(inj_weight) == len(inj_param)
   
    #Initialize variables/arrays
    ROC_value = 0
    FAP = []
    np.array(FAP)
    ROC_sum = []
    np.array(ROC_sum)

    for idx in range(len(noise_param)):
        #Calculate false alarm probability value
        FAP.append((float(idx+1))/len(noise_param))
        
        #Compute sum
        ROC_value = inj_weight[inj_param >= noise_param[idx]].sum()
        
        #Append
        ROC_sum.append(ROC_value)

        #Normalize ROC y axis
        ROC_sum = np.asarray(ROC_sum)
        ROC_sum *= (1.0/ROC_sum.max())

                
    return ROC_sum, FAP

#Calculate the ROC values
#ROC_w_sum, FAP = ROC(n_noise, prob_sort_injWeight, prob_sort_inj, prob_sort_noise)
#ROC_newsnr_sum, FAP = ROC(n_noise, newsnr_sort_injWeight, newsnr_sort_injNewsnr, newsnr_sort_noiseNewsnr)

#Create numpy arrays for FAP and ROC y-axis sums
FAP = []
ROC_w_sum = []
ROC_newsnr_sum = []
ROC_newsnr = []
np.array(FAP)
np.array(ROC_w_sum)
np.array(ROC_newsnr_sum)
np.array(ROC_newsnr)

#Initilize values for weighted sum and newsnr sum
w_sum = 0
newsnr_sum = 0

#Calculating the yaxis and FAP of ROC curve
for idx in range(n_noise):
    #Calculate false alarm probability value
    FAP.append((float(idx+1))/n_noise)

    w_sum = prob_sort_injWeight[prob_sort_inj >= prob_sort_noise[idx]].sum()  
    newsnr_sum = newsnr_sort_injWeight[newsnr_sort_injNewsnr >= newsnr_sort_noiseNewsnr[idx]].sum()  
    
    ROC_w_sum.append(w_sum)
    ROC_newsnr_sum.append(newsnr_sum)

#Normalizing ROC y axis
ROC_w_sum = np.asarray(ROC_w_sum)
ROC_w_sum *= (1.0/ROC_w_sum.max())

ROC_newsnr_sum = np.asarray(ROC_newsnr_sum)
ROC_newsnr_sum *= (1.0/ROC_newsnr_sum.max())

#Make Plots
os.makedirs('%s/run_%s' % (out_dir,now))
os.makedirs('%s/run_%s/colored_plots' % (out_dir,now))

#Saving model to hdf file for later use
model.save('%s/run_%s/nn_model.hdf' % (out_dir,now))
np.save('%s/run_%s/hist.npy' % (out_dir,now), hist.history)

#Make ROC Curve
pl.plot(FAP,ROC_w_sum,label='NN Score')
pl.plot(FAP,ROC_newsnr_sum,label='New SNR')
pl.legend(frameon=True)
pl.title('ROC Curve')
pl.xlabel('False Alarm Probability')
pl.ylabel('Weighted Sum')
pl.xscale('log')
pl.savefig('%s/run_%s/ROC_curve.png' % (out_dir,now))
pl.close()

#Make score vs. marginal likelihood plot
pl.scatter(test_data_p[0:len(trig_test),0],pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,0],pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Score vs. Marginal Likelihood')
pl.ylabel('Score')
pl.xlabel('Marginal Likelihood')
pl.savefig('%s/run_%s/score_vs_marg_l.png' % (out_dir,now))
pl.close()

#Make score vs. count plot
pl.scatter(test_data_p[0:len(trig_test),1],pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,1],pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Score vs. Count')
pl.ylabel('Score')
pl.xlabel('Count')
pl.savefig('%s/run_%s/score_vs_count.png' % (out_dir,now))
pl.close()

#Make score vs maxnewsnr plot
pl.scatter(test_data_p[0:len(trig_test),2],pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,2],pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Score vs. Max New SNR')
pl.ylabel('Score')
pl.xlabel('Max New SNR')
pl.savefig('%s/run_%s/score_vs_maxnewsnr.png' % (out_dir,now))
pl.close()

#Make score vs maxsnr plot
pl.scatter(test_data_p[0:len(trig_test),3],pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,3],pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Score vs. Maximum SNR')
pl.ylabel('Score')
pl.xlabel('Maximum SNR')
pl.savefig('%s/run_%s/score_vs_maxsnr.png' % (out_dir,now))
pl.close()

#Make score vs Ratio Chirp plot
pl.scatter(test_data_p[0:len(trig_test),4],pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,4],pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Score vs. Chirp Mass Ratio')
pl.ylabel('Score')
pl.xlabel('Chirp Mass Ratio')
pl.savefig('%s/run_%s/score_vs_ratio_chirp.png' % (out_dir,now))
pl.close()

#Make score vs time diffplot
pl.scatter(test_data_p[0:len(trig_test),5],pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,5],pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Score vs. Time Diff')
pl.ylabel('Score')
pl.xlabel('Time Diff')
pl.savefig('%s/run_%s/score_vs_delT.png' % (out_dir,now))
pl.close()

#Make score vs template duration diffplot
pl.scatter(test_data_p[0:len(trig_test),6],pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,6],pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Score vs. Template Duration')
pl.ylabel('Score')
pl.xlabel('Template Duration')
pl.savefig('%s/run_%s/score_vs_tmp_dur.png' % (out_dir,now))
pl.close()

#Colored Plots
#Count vs. marg_l plot
pl.scatter(test_data_p[0:len(trig_test),0],test_data_p[0:len(trig_test),1],c=pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,0],test_data_p[len(trig_test):,1],c=pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Count vs. Marg_l')
pl.xlabel('Marginal likelihood')
pl.ylabel('Count')
pl.colorbar()
pl.savefig('%s/run_%s/colored_plots/count_vs_marg_l.png' % (out_dir,now))
pl.close()

#Count vs. Maxnewsnr
pl.scatter(test_data_p[0:len(trig_test),2],test_data_p[0:len(trig_test),1],c=pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,2],test_data_p[len(trig_test):,1],c=pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Count vs. MaxNewSNR')
pl.xlabel('Max New SNR')
pl.ylabel('Count')
pl.colorbar()
pl.savefig('%s/run_%s/colored_plots/count_vs_maxnewsnr.png' % (out_dir,now))
pl.close()

#Count vs. Maxsnr
pl.scatter(test_data_p[0:len(trig_test),3],test_data_p[0:len(trig_test),1],c=pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,3],test_data_p[len(trig_test):,1],c=pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Count vs. MaxSNR')
pl.xlabel('Max SNR')
pl.ylabel('Count')
pl.colorbar()
pl.savefig('%s/run_%s/colored_plots/count_vs_maxsnr.png' % (out_dir,now))
pl.close()

#log(template duration) vs. New SNR
pl.scatter(test_data_p[0:len(trig_test),2],test_data_p[0:len(trig_test),6],c=pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,2],test_data_p[len(trig_test):,6],c=pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('log Template Duration vs. New SNR')
pl.xlabel('New SNR')
pl.ylabel('log Template Duration')
pl.colorbar()
pl.savefig('%s/run_%s/colored_plots/tempdur_vs_maxnewsnr.png' % (out_dir,now))
pl.close()

#Marg_l vs. Maxnewsnr
pl.scatter(test_data_p[0:len(trig_test),2],test_data_p[0:len(trig_test),0],c=pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,2],test_data_p[len(trig_test):,0],c=pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Marg_l vs. MaxNewSNR')
pl.xlabel('Max New SNR')
pl.ylabel('Marginal Likelihood')
pl.colorbar()
pl.savefig('%s/run_%s/colored_plots/marg_l_vs_maxnewsnr.png' % (out_dir,now))
pl.close()

#Marg_l vs. MaxSNR
pl.scatter(test_data_p[0:len(trig_test),3],test_data_p[0:len(trig_test),0],c=pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,3],test_data_p[len(trig_test):,0],c=pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('Marg_l vs. MaxSNR')
pl.xlabel('Max SNR')
pl.ylabel('Marginal Likelihood')
pl.colorbar()
pl.savefig('%s/run_%s/colored_plots/marg_l_vs_maxsnr.png' % (out_dir,now))
pl.close()

#MaxNewSNR vs. MaxSNR
pl.scatter(test_data_p[0:len(trig_test),3],test_data_p[0:len(trig_test),2],c=pred_prob[0:len(trig_test)],marker="o",label='background')
pl.scatter(test_data_p[len(trig_test):,3],test_data_p[len(trig_test):,2],c=pred_prob[len(trig_test):],marker="^",label='injection')
pl.legend(frameon=True)
pl.title('MaxNewSNR vs. MaxSNR')
pl.xlabel('Max SNR')
pl.ylabel('Max New SNR')
pl.colorbar()
pl.savefig('%s/run_%s/colored_plots/maxnewsnr_vs_maxsnr.png' % (out_dir,now))
pl.close()

#Loss vs. Epoch
pl.plot(hist.history['loss'])
pl.title('Loss vs. Epoch')
pl.xlabel('Epoch')
pl.ylabel('Loss')
pl.savefig('%s/run_%s/loss_vs_epoch.png' % (out_dir,now))

#Accuracy vs. Epoch
pl.plot(hist.history['acc'])
pl.title('Accuracy vs. Epoch')
pl.xlabel('Epoch')
pl.ylabel('Accuracy')
pl.savefig('%s/run_%s/acc_vs_epoch.png' % (out_dir,now))

#Write data to an hdf file
with h5py.File('nn_data.hdf', 'w') as hf:
    hf.create_dataset('FAP', data=FAP)
    hf.create_dataset('ROC_w_sum', data=ROC_w_sum)
    hf.create_dataset('pred_prob', data=pred_prob)
    hf.create_dataset('test_data', data=test_data)
    hf.create_dataset('train_data', data=train_data)
    hf.create_dataset('ROC_newsnr_sum', data=ROC_newsnr_sum)
