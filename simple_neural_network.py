#Simple Multilayer Neural Network to seperate pycbc injections from noise triggers
#Author: Hunter Gabbard
#Max Planck Insitute for Gravitational Physics
#How to use only one GPU device....export CUDA_VISIBLE_DEVICES="0" in command line prior to run
#How to run in the background from the command line...python simple_neural_network.py -d NSBH01_ifar0-1.hdf,NSBH02_ifar0-1.hdf >/dev/null 2>err.txt &

from __future__ import division
import argparse
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
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
args = vars(ap.parse_args())

data_files = args['dataset'].split(',')

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

    #Applying weights to injections
    marg_l_inj_w = []
    count_inj_w = []
    maxnewsnr_inj_w = []
    maxsnr_inj_w = []
    time_inj_w = []
    eff_dist_inj_w = []
    ratio_chirp_inj_w = []
    delT_inj_w = []
    delta_chirp_inj_w = []

inj_weights_pre = []
np.asarray(inj_weights_pre)
dist_inj_mean = (eff_dist_inj**2).mean()
for idx in enumerate(delta_chirp_inj):
    idx = idx[0]
    inj_weights_pre.append((eff_dist_inj[idx][0]**2)/dist_inj_mean)

inj_weights = np.asarray(inj_weights_pre).reshape((delta_chirp_inj.shape[0],1))

#Combining injection parameters into a numpy array
inj_comb = np.hstack((marg_l_inj, count_inj, maxnewsnr_inj, maxsnr_inj, ratio_chirp_inj, delT_inj))

#Randomizing the order of the background triggers
indices_trig = np.random.permutation(trig_comb.shape[0])

#Seperate data into training and testing
trig_train_idx, trig_test_idx = indices_trig[:int(trig_comb.shape[0]*.7)], indices_trig[int(trig_comb.shape[0]*.7):int(trig_comb.shape[0])]
trig_train, trig_test = trig_comb[trig_train_idx,:], trig_comb[trig_test_idx,:]
indices_inj = np.random.permutation(inj_comb.shape[0])
inj_train_idx, inj_test_idx = indices_inj[:int(inj_comb.shape[0]*.7)], indices_inj[int(inj_comb.shape[0]*.7):]
inj_train_weight, inj_test_weight = inj_weights[inj_train_idx,:], inj_weights[inj_test_idx,:]
inj_train, inj_test = inj_comb[inj_train_idx,:], inj_comb[inj_test_idx,:]
comb_all = np.vstack((trig_comb, inj_comb))
train_data = np.vstack((trig_train, inj_train))
test_data = np.vstack((trig_test, inj_test))


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
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.add(Dense(1, input_dim=trig_comb.shape[1],activation='relu'))
#model.add(Dense(1, input_dim=1,activation='linear'))

#model.add(Dense(300, activation='sigmoid'))
#model.add(Dropout(0.2))
#model.add(Dense(500, activation='sigmoid'))
#model.add(Dropout(0.2))
#model.add(Dense(700, activation='sigmoid'))
#model.add(Dropout(0.2))
#model.add(Dense(500, activation='sigmoid'))
#model.add(Dropout(0.2))
#model.add(Dense(200, activation='sigmoid'))

model.add(Dense(1, activation='sigmoid'))

#Compiling model
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer='sgd',
	metrics=["accuracy"], class_mode='binary')

#Creating sample weights vector
trig_weights = np.zeros((trig_comb.shape[0],1))
trig_weights.fill(1/((trig_comb.shape[0])/(inj_comb.shape[0])))
trig_w_train = trig_weights[:trig_comb.shape[0]*.7] 
trig_w_test = trig_weights[trig_comb.shape[0]*.7:]
train_weights = np.vstack((trig_w_train,inj_train_weight)).flatten()

model.fit(train_data, lab_train, nb_epoch=1, batch_size=32, sample_weight=train_weights, shuffle=True, show_accuracy=True)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(test_data, lab_test,
	batch_size=32, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))
#Saving prediction probabilities to a variable
res_pre = model.predict(test_data)

#Printing summary of model parameters
model.summary()

#Saving model to hdf file for later use
model.save('nn_model.hdf')

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
#def ROC(n_noise, weight, inj_param, noise_param):

#    ROC_value = 0
#    FAP = []
#    np.array(FAP)
#    ROC_sum = []
#    np.array(ROC_sum)

#    for idx in range(n_noise):
        #Calculate false alarm probability value
#        FAP.append((float(idx+1))/n_noise)
        
        #Compute sum
#        ROC_value = weight[inj_param >= noise_param[idx]].sum()
        
        #Append
#        ROC_sum = np.append(ROC_sum, ROC_value)

        #Normalize ROC y axis
#        ROC_sum = np.asarray(ROC_sum)
#        ROC_sum *= (1.0/ROC_sum.max())

                
#    return ROC_sum, FAP

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
os.makedirs('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s' % now)
os.makedirs('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/colored_plots' % now)

#Make ROC Curve
pl.plot(FAP,ROC_w_sum,label='NN Score')
pl.plot(FAP,ROC_newsnr_sum,label='New SNR')
pl.legend()
pl.title('ROC Curve')
pl.xlabel('False Alarm Probability')
pl.ylabel('Weighted Sum')
pl.xscale('log')
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/ROC_curve.png' % now)
pl.close()

#Make Marginal liklihood vs. score plot
pl.scatter(pred_prob,test_data[:,0])
pl.title('Marginal Likelihood vs. Score')
pl.xlabel('Score')
pl.ylabel('Marginal Likelihood')
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/marg_l_vs_score.png' % now)
pl.close()

#Make Count vs score plot
pl.scatter(pred_prob,test_data[:,1])
pl.title('Count vs. Score')
pl.xlabel('Score')
pl.ylabel('Count')
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/count_vs_score.png' % now)
pl.close()

#Make maxnewsnr vs score plot
pl.scatter(pred_prob,test_data[:,2])
pl.title('Max New SNR vs. Score')
pl.xlabel('Score')
pl.ylabel('Max New SNR')
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/maxnewsnr_vs_score.png' % now)
pl.close()

#Make maxsnr vs score plot
pl.scatter(pred_prob,test_data[:,3])
pl.title('Maximum SNR vs. Score')
pl.xlabel('Score')
pl.ylabel('Maximum SNR')
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/maxsnr_vs_score.png' % now)
pl.close()

#Make Ratio Chirp vs score plot
pl.scatter(pred_prob,test_data[:,4])
pl.title('Chirp Mass Ratio vs. Score')
pl.xlabel('Score')
pl.ylabel('Chirp Mass Ratio')
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/ratio_chirp_vs_score.png' % now)
pl.close()

#Make time diff vs score plot
pl.scatter(pred_prob,test_data[:,0])
pl.title('Time Diff vs. Score')
pl.xlabel('Score')
pl.ylabel('Time Diff')
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/delT_vs_score.png' % now)
pl.close()

#Colored Plots
#Count vs. marg_l plot
pl.scatter(test_data[:,0],test_data[:,1],c=pred_prob)
pl.title('Count vs. Marg_l')
pl.colorbar()
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/colored_plots/count_vs_marg_l.png' % now)
pl.close()

#Count vs. Maxnewsnr
pl.scatter(test_data[:,2],test_data[:,1],c=pred_prob)
pl.title('Count vs. MaxNewSNR')
pl.colorbar()
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/colored_plots/count_vs_maxnewsnr.png' % now)
pl.close()

#Count vs. Maxsnr
pl.scatter(test_data[:,3],test_data[:,1],c=pred_prob)
pl.title('Count vs. MaxSNR')
pl.colorbar()
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/colored_plots/count_vs_maxsnr.png' % now)
pl.close()

#Marg_l vs. Maxnewsnr
pl.scatter(test_data[:,2],test_data[:,0],c=pred_prob)
pl.title('Marg_l vs. MaxNewSNR')
pl.colorbar()
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/colored_plots/marg_l_vs_maxnewsnr.png' % now)
pl.close()

#Marg_l vs. MaxSNR
pl.scatter(test_data[:,3],test_data[:,0],c=pred_prob)
pl.title('Marg_l vs. MaxSNR')
pl.colorbar()
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/colored_plots/marg_l_vs_maxsnr.png' % now)
pl.close()

#MaxNewSNR vs. MaxSNR
pl.scatter(test_data[:,3],test_data[:,2],c=pred_prob)
pl.title('MaxNewSNR vs. MaxSNR')
pl.colorbar()
pl.savefig('/home/hunter.gabbard/public_html/simple_neural_net/inj_v_back_stats/run_%s/colored_plots/maxnewsnr_vs_maxsnr.png' % now)
pl.close()

#Write data to an hdf file
with h5py.File('nn_data.hdf', 'w') as hf:
    hf.create_dataset('FAP', data=FAP)
    hf.create_dataset('ROC_w_sum', data=ROC_w_sum)
    hf.create_dataset('pred_prob', data=pred_prob)
    hf.create_dataset('test_data', data=test_data)
    hf.create_dataset('train_data', data=train_data)
    hf.create_dataset('ROC_newsnr_sum', data=ROC_newsnr_sum)
