# Copyright (C) 2017  Hunter A. Gabbard
# Most of this has been shamelessly copied from Duncan Mcleod's GWPY qtransform.py script
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

"""
This module retrives a timeseries and then calculates the q-transform of that time series
"""

from math import pi, ceil, log, exp
import numpy as np
from pycbc.types.timeseries import FrequencySeries, TimeSeries
import os, sys
from pycbc.frame import read_frame
from pycbc.filter import highpass_fir, matched_filter
from pycbc.waveform import get_fd_waveform
from pycbc.psd import welch, interpolate
from pycbc.fft import ifft
import urllib
import datetime
from scipy.signal import tukey
from numpy import fft as npfft
from matplotlib import use
use('Agg')
from matplotlib import pyplot as pl
import argparse
import datetime

__author__ = 'Hunter Gabbard <hunter.gabbard@ligo.org>'
__credits__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

def qtiling(h1, qrange, frange, sampling, normalized, mismatch):
    """
    Parameters
    """
    deltam = deltam_f(mismatch)
    qrange = (float(qrange[0]), float(qrange[1]))
    frange = [float(frange[0]), float(frange[1])]
    dur = int(len(h1)) / sampling # length of your data chunk in seconds ... self.duration
    qplane_tile_dict = {}

    qs = list(_iter_qs(qrange, deltam))
    if frange[0] == 0:  # set non-zero lower frequency
        frange[0] = 50 * max(qs) / (2 * pi * dur)
    if np.isinf(frange[1]):  # set non-infinite upper frequency
        frange[1] = sampling / 2 / (1 + 11**(1/2.) / min(qs))

    #lets now define the whole tiling (e.g. choosing all tiling in planes)
    for q in qs:
        qtilefreq = np.array(list(_iter_frequencies(q, frange, mismatch, dur)))
        qlst = np.empty(len(qtilefreq), dtype=float)
        qlst.fill(q)
        qtiles_array = np.vstack((qtilefreq,qlst)).T
        qplane_tiles_list = list(map(tuple,qtiles_array))
        qplane_tile_dict[q] = qplane_tiles_list 

    #perform q-transform on each tile for each q-plane and pick out the tile that has the largest normalized energy 
    #store q-transforms of each tile in a dict
    qplane_qtrans_dict = {}
 
    for i, key in enumerate(qplane_tile_dict):
        print key
        norm_energies_lst=[]
        for tile in qplane_tile_dict[key]:
            norm_energies = qtransform(h1, tile[1], tile[0], sampling, normalized)
            norm_energies_lst.append(norm_energies)
            if i == 0:
                max_norm_energy = max(norm_energies)
            elif max(norm_energies) > max_norm_energy:
                max_norm_energy = max(norm_energies)

        qplane_qtrans_dict[key] = np.array(norm_energies_lst)
    print max_norm_energy
    sys.exit() 


def deltam_f(mismatch):
    """Fractional mismatch between neighbouring tiles
    :type: `float`
    """
    return 2 * (mismatch / 3.) ** (1/2.)


def _iter_qs(qrange, deltam):
    """Iterate over the Q values
    """

    # work out how many Qs we need
    cumum = log(qrange[1] / qrange[0]) / 2**(1/2.)
    nplanes = int(max(ceil(cumum / deltam), 1))
    dq = cumum / nplanes
    for i in xrange(nplanes):
        yield qrange[0] * exp(2**(1/2.) * dq * (i + .5))
    raise StopIteration()

def _iter_frequencies(q, frange, mismatch, dur):
    """Iterate over the frequencies of this `QPlane`
    """
    # work out how many frequencies we need
    minf, maxf = frange
    fcum_mismatch = log(maxf / minf) * (2 + q**2)**(1/2.) / 2.
    nfreq = int(max(1, ceil(fcum_mismatch / deltam_f(mismatch))))
    fstep = fcum_mismatch / nfreq
    fstepmin = 1 / dur
    # for each frequency, yield a QTile
    for i in xrange(nfreq):
        yield (minf *
               exp(2 / (2 + q**2)**(1/2.) * (i + .5) * fstep) //
               fstepmin * fstepmin)
    raise StopIteration()

def qtransform(data, Q, f0, sampling, normalized):

    """
    Parameters
    ----------
    data : `LIGO gwf frame file`
        raw time-series data set
    normalized : `bool`, optional
        normalize the energy of the output, if `False` the output
        is the complex `~numpy.fft.ifft` output of the Q-tranform
    f0 :
        central frequency
    sampling :
        sampling frequency of channel
    normalized:
        normalize output tile energies? 
    """

    #Q-transform data for each (Q, frequency) tile

    #Initialize parameters
    qprime = Q / 11**(1/2.) # ... self.qprime
    dur = int(len(data)) / sampling # length of your data chunk in seconds ... self.duration
    print 'starting pycbc fft ...'
    fseries = TimeSeries.to_frequencyseries(data)
      
    #Window fft
    window_size = 2 * int(f0 / qprime * dur) + 1
        
    #Get indices
    indices = _get_indices(dur)

    #Apply window to fft
    windowed = fseries[get_data_indices(dur, f0, indices)] * get_window(dur, indices, f0, qprime, Q, sampling)

    # pad data, move negative frequencies to the end, and IFFT
    padded = np.pad(windowed, padding(window_size, dur, f0, Q), mode='constant')
    print 'starting numpy ifftshift'
    wenergy = npfft.ifftshift(padded)

    # return a `TimeSeries`
    wenergy = FrequencySeries(wenergy, delta_f=sampling)
    print 'doing pycbc ifft'
    tdenergy = FrequencySeries.to_timeseries(wenergy)
    cenergy = TimeSeries(tdenergy,
                         delta_t=1, copy=False) # Normally delta_t is dur/tdenergy.size ... must figure out better way of doing this
    if normalized:
        energy = type(cenergy)(
            cenergy.real() ** 2. + cenergy.imag() ** 2.,
            delta_t=1, copy=False)
        meanenergy = energy.numpy().mean()
        result = energy / meanenergy
    else:
        result = cenergy
   
    return result

def padding(window_size, dur, f0, Q):
        """The `(left, right)` padding required for the IFFT
        :type: `tuple` of `int`
        """
        pad = n_tiles(dur,f0,Q) - window_size
        return (int((pad - 1)/2.), int((pad + 1)/2.))

def get_data_indices(dur, f0, indices):
    """Returns the index array of interesting frequencies for this row
    """
    return np.round(indices + 1 +
                       f0 * dur).astype(int)

def _get_indices(dur):
    half = int((int(dur) - 1.) / 2.) 
    return np.arange(-half, half + 1)

def get_window(dur, indices, f0, qprime, Q, sampling):
    """Generate the bi-square window for this row
    Returns
    -------
    window : `numpy.ndarray`
    """
    # real frequencies
    wfrequencies = indices / dur

    # dimensionless frequencies
    xfrequencies = wfrequencies * qprime / f0

    # normalize and generate bi-square window
    norm = n_tiles(dur,f0,Q) / (dur * sampling) * (
        315 * qprime / (128 * f0)) ** (1/2.)
    return (1 - xfrequencies ** 2) ** 2 * norm

def n_tiles(dur,f0,Q):
    """The number of tiles in this row 

    :type: 'int'
    """
    

    tcum_mismatch = dur * 2 * pi * f0 / Q 
    return next_power_of_two(tcum_mismatch / deltam())

def next_power_of_two(x):
    """Return the smallest power of two greater than or equal to `x`
    """
    return 2**(ceil(log(x, 2)))

def deltam():
    """Fractional mismatch between neighbouring tiles
    :type: `float`
    """
    mismatch = 0.2
    return 2 * (mismatch / 3.) ** (1/2.)

def main():
    #Get Current time
    cur_time = datetime.datetime.now()

    #construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--usertag", required=False, default=cur_time,
        help="label for given run")
    ap.add_argument("-o", "--output-dir", required=False,
        help="path to output directory")
    ap.add_argument("-n", "--normalize", required=False, default=True,
        help="normalize the energy of the output")
    ap.add_argument("-s", "--samp-freq", required=True, type=float,
        help="Sampling frequency of channel")

    args = ap.parse_args()


    #Initialize parameters
    out_dir = args.output_dir
    now = args.usertag
    #os.makedirs('%s/run_%s' % (out_dir,now))  # Fail early if the dir already exists
    normalized = args.normalize # Set this as needed
    Q = 20 # self-explanatory ... self.q
    f0 = 5 # initiail frequency ... self.frequency
    sampling = args.samp_freq #sampling frequency
    mismatch=.2
    qrange=(4,64)
    frange=(0,np.inf)

    # Read data and remove low frequency content
    fname = 'H-H1_LOSC_4_V2-1126259446-32.gwf'
    url = "https://losc.ligo.org/s/events/GW150914/" + fname
    urllib.urlretrieve(url, filename=fname)
    h1 = read_frame('H-H1_LOSC_4_V2-1126259446-32.gwf', 'H1:LOSC-STRAIN')
    h1 = TimeSeries(np.random.normal(size=256*4096), delta_t = 1. / sampling)
    h1 = highpass_fir(h1, 15, 8)

    # Calculate the noise spectrum
    psd = interpolate(welch(h1), 1.0 / 32)

    #perform Q-tiling
    Qbase = qtiling(h1, qrange, frange, sampling, normalized, mismatch)

    #FOR DIAGNOSTIC PURPOSES ONLY
    #pl.figure()
    #numpy_hist, bins = np.histogram(q, bins=100, density=True)
    #width = (bins[1] - bins[0])
    #center = (bins[:-1] + bins[1:]) / 2
    #pl.bar(center, numpy_hist, log=True, label='normalised energies', color='r', alpha=0.4, align='center', width=width)
    #pl.legend(frameon=True)
    #pl.savefig('%s/run_%s/energies_hist.png' % (out_dir,now))
    #pl.close()
 

if __name__ == '__main__':
    main()
