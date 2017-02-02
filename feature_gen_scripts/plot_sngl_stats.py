
from matplotlib import use
use('Agg')
from matplotlib import pyplot as pl
import numpy as np
import h5py, argparse
from numpy import exp

parser = argparse.ArgumentParser()
parser.add_argument('--stat-file')
parser.add_argument('--ifo')
parser.add_argument('--output-dir')
parser.add_argument('--user-tag', default="")

args = parser.parse_args()
out = args.output_dir
ifo = args.ifo
tag = args.user_tag


#Downloading all triggers
from pycbc import events, init_logging, pnutils
hf_all = h5py.File('H1-HDF_TRIGGER_MERGE_NSBH02_INJ-1128299417-1083600.hdf', 'r')
f_all = hf_all['H1']
time_all = f_all['end_time'][:]
snr_all = f_all['snr'][:]
chisq = f_all['chisq'][:]
chisq_dof = f_all['chisq_dof'][:]
rchisq = chisq / (2 * chisq_dof - 2)
del chisq
del chisq_dof
newsnr_all = events.newsnr(snr_all, rchisq)
del rchisq



f = h5py.File(args.stat_file, 'r')
newsnr = f['%s/maxnewsnr' % ifo][:]
maxsnr = f['%s/maxsnr' % ifo][:]
time = f['%s/time' % ifo][:]
time_inj = f['%s/time_inj' % ifo][:]
count = f['%s/count' % ifo][:]
margl = f['%s/marg_l' % ifo][:]
delT = f['%s/delT' % ifo][:]
delta_chirp = f['%s/delta_chirp' % ifo][:]
newsnr_inj = f['%s/maxnewsnr_inj' % ifo][:]
maxsnr_inj = f['%s/maxsnr_inj' % ifo][:]
count_inj = f['%s/count_inj' % ifo][:]
margl_inj = f['%s/marg_l_inj' % ifo][:]
delT_inj = f['%s/delT_inj' % ifo][:]
delta_chirp_inj = f['%s/delta_chirp_inj' % ifo][:]
ratio_chirp = f['%s/ratio_chirp' % ifo][:]
ratio_chirp_inj = f['%s/ratio_chirp_inj' % ifo][:]

#Triggers where max new snr is greater than 10
newsnr_l10 = newsnr[np.log(newsnr) < 10]
newsnr_inj_l10 = newsnr_inj[np.log(newsnr_inj) < 10]
maxsnr_l10 = maxsnr[np.log(newsnr) < 10]
maxsnr_inj_l10 = maxsnr_inj[np.log(newsnr_inj) < 10]
delT_l10 = delT[np.log(newsnr) < 10]
delT_inj_l10 = delT_inj[np.log(newsnr_inj) < 10]
#pl.plot(maxsnr/newsnr, delT, 'k.', ms=2)
#pl.plot(maxsnr_inj_l10/newsnr_inj_l10, delT_inj_l10,'r.')
pl.plot(np.log(newsnr_inj), maxsnr_inj/newsnr_inj, 'r.')
#pl.xscale('log')
#pl.xlim(0.9*min((maxsnr/newsnr).min(), (maxsnr_inj_l10/newsnr_inj_l10).min()), 1.1*max((maxsnr/newsnr).max(), (maxsnr_inj_l10/newsnr_inj_l10).max()))
#pl.ylim(0.7*min((delT).min(), (delT_inj_l10).min()), 1.4*max((delT_inj_l10).max(), (delT).max()))
pl.xlabel('Max SNR / max newsnr')
pl.ylabel('Time Difference (max SNR - max newSNR)')
pl.grid(True)
pl.savefig('%s/%s-delT_v_maxsnr_norm_l10_%s.png' % (out, ifo, tag))
pl.close()

#Determine where this outlier injection is
#time_inj_l10 = time_inj[np.log(newsnr_inj) < 10]
newsnr_inj2 = newsnr_inj[newsnr_inj < 10]
maxsnr_inj2 = maxsnr_inj[newsnr_inj < 10]
maxsnr_inj_tom = maxsnr_inj2[maxsnr_inj2 > 1.2*newsnr_inj2] #[delT_inj > 0.0000001]
newsnr_inj_tom = newsnr_inj2[maxsnr_inj2 > 1.2*newsnr_inj2] #[delT_inj > 0.0000001]
time_inj2 = time_inj[newsnr_inj < 10] #[delT_inj > 0.0000001]
delT_inj2 = delT_inj[newsnr_inj < 10]
time_inj_l10 = time_inj2[maxsnr_inj2 > 1.2*newsnr_inj2]
delT_inj_l10 = delT_inj2[maxsnr_inj2 > 1.2*newsnr_inj2]

print time_inj_l10
print delT_inj_l10

idx_injratio = np.argmax((maxsnr_inj_l10/newsnr_inj_l10))

#print time_inj_l10[idx_injratio]

#artifact_time = time_all[abs(time_inj_l10[idx_injratio] - time_all) < 2] 
#artifact_snr = snr_all[abs(time_inj_l10[idx_injratio] - time_all) < 2]
#artifact_newsnr = newsnr_all[abs(time_inj_l10[idx_injratio] - time_all) < 2]

inj_art_time = time_all[abs(time_inj_l10[0] - time_all) < 2]
inj_art_snr = snr_all[abs(time_inj_l10[0] - time_all) < 2]
inj_art_newsnr = newsnr_all[abs(time_inj_l10[0] - time_all) < 2]

pl.plot(inj_art_time, inj_art_snr, 'r.')
#pl.xlim(0.9*artifact_time.min(), 1.1*artifact_time.max())
#pl.ylim(0.7*artifact_snr.min(), 1.4*artifact_snr.max())
pl.xlabel('Time')
pl.ylabel('SNR')
pl.grid(True)
pl.savefig('%s/%s-snr_v_time_%s_trig1.png' % (out, ifo, tag))
pl.close()

pl.plot(inj_art_time, inj_art_newsnr, 'r.')
#pl.xlim(0.9*artifact_time.min(), 1.1*artifact_time.max())
#pl.ylim(0.7*artifact_newsnr.min(), 1.4*artifact_newsnr.max())
pl.xlabel('Time')
pl.ylabel('New SNR')
pl.grid(True)
pl.savefig('%s/%s-newsnr_v_time_%s_trig1.png' % (out, ifo, tag))
pl.close()




exit()


########################################################################################################################################
pl.loglog(newsnr, maxsnr, 'k.', ms=2)
pl.loglog(newsnr_inj, maxsnr_inj, 'r.')
#pl.loglog(newsnr_inj, 1.2*newsnr_inj, 'b')
#pl.loglog(newsnr_inj, np.poly1d(np.polyfit(newsnr_inj, maxsnr_inj, 5)), 'b')
pl.xlim(0.9*min(newsnr.min(), newsnr_inj.min()), 1.1*max(newsnr_inj.max(), newsnr.max()))
pl.ylim(0.7*min(maxsnr.min(), maxsnr_inj.min()), 1.4*max(maxsnr.max(), maxsnr_inj.max()))
pl.xlabel('Maximum Reweighted SNR')
pl.ylabel('Maximum SNR')
pl.grid(True)
pl.savefig('%s/%s-maxsnr_v_newsnr_%s.png' % (out, ifo, tag))
pl.close()

pl.loglog(newsnr, maxsnr/newsnr, 'k.', ms=2)
pl.loglog(newsnr_inj, maxsnr_inj/newsnr_inj, 'r.')
#pl.loglog(newsnr_inj, 1.2*np.ones_like(newsnr_inj), 'b')
pl.xlim(0.9*min(newsnr.min(), newsnr_inj.min()), 1.1*max(newsnr_inj.max(), newsnr.max()))
pl.ylim(0.7*min((maxsnr/newsnr).min(), (maxsnr_inj/newsnr_inj).min()), 1.4*max((maxsnr/newsnr).max(), (maxsnr_inj/newsnr_inj).max()))
pl.xlabel('Maximum Reweighted SNR')
pl.ylabel('Max SNR / max newsnr')
pl.grid(True)
pl.savefig('%s/%s-maxsnr_norm_v_newsnr_%s.png' % (out, ifo, tag))
pl.close()

pl.plot(maxsnr/newsnr, delT, 'k.', ms=2)
pl.plot(maxsnr_inj/newsnr_inj, delT_inj, 'r.')
pl.xscale('log')
pl.xlim(0.9*min((maxsnr/newsnr).min(), (maxsnr_inj/newsnr_inj).min()), 1.1*max((maxsnr/newsnr).max(), (maxsnr_inj/newsnr_inj).max()))
pl.ylim(0.7*min((delT).min(), (delT_inj).min()), 1.4*max((delT_inj).max(), (delT).max()))
pl.xlabel('Max SNR / max newsnr')
pl.ylabel('Time Difference (max SNR - max newSNR)')
pl.grid(True)
pl.savefig('%s/%s-delT_v_maxsnr_norm_%s.png' % (out, ifo, tag))
pl.close()

pl.plot(newsnr, delT, 'k.', ms=2)
pl.plot(newsnr_inj, delT_inj, 'r.')
pl.xscale('log')
pl.xlim(0.9*min(newsnr.min(), newsnr_inj.min()), 1.1*max(newsnr_inj.max(), newsnr.max()))
pl.ylim(0.9*min(delT.min(), delT_inj.min()), 1.1*max(delT_inj.max(), delT.max()))
pl.xlabel('Maximum Reweighted SNR')
pl.ylabel('Time difference (max SNR - max newSNR)')
pl.grid(True)
pl.savefig('%s/%s-delT_v_newsnr_%s.png' % (out, ifo, tag))
pl.close()

pl.loglog(newsnr, ratio_chirp, 'k.', ms=2)
pl.loglog(newsnr_inj, ratio_chirp_inj, 'r.')
pl.xlim(0.9*min(newsnr.min(), newsnr_inj.min()), 1.1*max(newsnr_inj.max(), newsnr.max()))
pl.ylim(0.9*min(ratio_chirp.min(), ratio_chirp_inj.min()), 1.1*max(ratio_chirp_inj.max(), ratio_chirp.max()))
pl.xlabel('Maximum Reweighted SNR')
pl.ylabel('Chirp mass ratio (max SNR/max newSNR)')
pl.grid(True)
pl.savefig('%s/%s-chirp_ratio_v_newsnr_%s.png' % (out, ifo, tag))
pl.close()

pl.loglog(newsnr, delta_chirp, 'k.', ms=2)
pl.loglog(newsnr_inj, delta_chirp_inj, 'r.')
pl.xlim(0.9*min(newsnr.min(), newsnr_inj.min()), 1.1*max(newsnr_inj.max(), newsnr.max()))
pl.ylim(0.9*min(delta_chirp.min(), delta_chirp_inj.min()), 1.1*max(delta_chirp_inj.max(), delta_chirp.max()))
pl.xlabel('Maximum Reweighted SNR')
pl.ylabel('Difference in Chirp Mass')
pl.grid(True)
pl.savefig('%s/%s-chirp_diff_v_newsnr_%s.png' % (out, ifo, tag))
pl.close()

pl.loglog(newsnr, abs(count), 'k.', ms=2)
pl.loglog(newsnr_inj, abs(count_inj), 'r.')
pl.xlim(0.9*min(newsnr.min(), newsnr_inj.min()), 1.1*max(newsnr_inj.max(), newsnr.max()))
pl.ylim(0.9*min(abs(count).min(), abs(count_inj).min()), 1.1*max(abs(count_inj).max(), abs(count).max()))
pl.xlabel('Maximum Reweighted SNR')
pl.ylabel('Count')
pl.grid(True)
pl.savefig('%s/%s-count_v_newsnr_%s.png' % (out, ifo, tag))
pl.close()

pl.loglog(newsnr, margl, 'k.', ms=2)
pl.loglog(newsnr_inj, margl_inj, 'r.')
pl.xlim(0.9*min(newsnr.min(), newsnr_inj.min()), 1.1*max(newsnr_inj.max(), newsnr.max()))
pl.ylim(0.9*min(margl.min(), margl_inj.min()), 1.1*max(margl_inj.max(), margl.max()))
pl.ylabel('Marg Likelihood')
pl.xlabel('Maximum Reweighted SNR')
pl.grid(True)
pl.savefig('%s/%s-margl_v_newsnr_%s.png' % (out, ifo, tag))
pl.close()

