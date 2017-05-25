# Copyright (C) 2017  Hunter A. Gabbard
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
from pycbc.types import zeros
import os, sys
from pycbc.frame import read_frame
from pycbc.filter import highpass_fir, matched_filter
from pycbc.waveform import get_fd_waveform
from pycbc.psd import welch, interpolate
from pycbc.fft import ifft
import urllib
import datetime
from scipy.interpolate import (interp2d, InterpolatedUnivariateSpline)
from numpy import fft as npfft
import argparse
import datetime

from matplotlib import use
use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import specgram

__author__ = 'Hunter Gabbard <hunter.gabbard@ligo.org>, Andrew Lundgren <andrew.lundgren@aei.mpg.de'
__credits__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

def plotter(interp, out_dir, now, frange, h1, sampling, tres, fres):
    """
    Parameters
    """

    # plot a spectrogram of the q-plane with the loudest normalized tile energy
    print 'plotting ...'    
    # intitialize variables
    xarr=np.linspace(0, 1, 1. / tres)
    yarr=np.arange(int(frange[0]), int(frange[1]), fres)
    z = interp(xarr,yarr)

    # pick the desired colormap
    cmap = plt.get_cmap('viridis')

    fig = plt.plot()

    p1 = plt.pcolormesh(xarr,
                        yarr,
                        z, cmap=cmap, norm=None)
    plt.colorbar(p1)
    plt.title('Pycbc q-transform')

    plt.savefig('%s/run_%s/spec.png' % (out_dir,now))

    return plt


def Qplane(qplane_tile_dict, h1, sampling, normalized, out_dir, now, frange, tres, fres):
    """
    Parameters
    """
    # perform q-transform on each tile for each q-plane and pick out tile with largest normalized energy
    # store q-transforms of each tile in a dict
    qplane_qtrans_dict = {}
    dur = int(len(h1)) / sampling

    max_norm_energy = []
    for i, key in enumerate(qplane_tile_dict):
        print key
        norm_energies_lst=[]
        for tile in qplane_tile_dict[key]:
            norm_energies = qtransform(h1, tile[1], tile[0], sampling, normalized)
            norm_energies_lst.append(norm_energies)
            if i == 0:
                max_norm_energy.append(max(norm_energies))
                max_norm_energy.append(tile)
                max_norm_energy.append(key)
            elif max(norm_energies) > max_norm_energy[0]:
                max_norm_energy[0] = max(norm_energies)
                max_norm_energy[1] = tile
                max_norm_energy[2] = key
                max_norm_energy[3] = norm_energies
        qplane_qtrans_dict[key] = np.array(norm_energies_lst)

    # record peak q calculate above and q-transform output for peak q
    peakq = max_norm_energy[1][1]
    result = qplane_qtrans_dict[max_norm_energy[2]]

    # then interpolate the spectrogram to increase the frequency resolution
    if fres is None:  # unless user tells us not to
        return result
    else:
        # initialize some variables
        frequencies = []

        for idx, i in enumerate(qplane_tile_dict[max_norm_energy[2]]):
            frequencies.append(i[0])

        # 2-D interpolation
        time_array = np.linspace(-(dur / 2.),(dur / 2.),int(dur * sampling))
        interp = interp2d(time_array, frequencies, result)

    out = interp(np.linspace(0, 1, 1. / tres), np.arange(int(frange[0]), int(frange[1]), fres))

    return out, interp


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

    return qplane_tile_dict, frange


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
    fstepmin = 1. / dur
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
    data : `pycbc TimeSeries`
        whitened time-series
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
    fseries = TimeSeries.to_frequencyseries(data)
      
    #Window fft
    window_size = 2 * int(f0 / qprime * dur) + 1
        
    #Get indices
    indices = _get_indices(window_size)

    #Apply window to fft
    windowed = fseries[get_data_indices(dur, f0, indices)] * get_window(window_size, f0, qprime)

    # Choice of output sampling rate
    output_sampling = sampling # Can lower this to highest bandwidth
    output_samples = dur * output_sampling

    # pad data, move negative frequencies to the end, and IFFT 
    padded = np.pad(windowed, padding(window_size, output_samples), mode='constant')
    wenergy = npfft.ifftshift(padded)

    # return a `TimeSeries`
    wenergy = FrequencySeries(wenergy, delta_f=1./dur)
    tdenergy = TimeSeries(zeros(output_samples, dtype=np.complex128),
                            delta_t=1./output_sampling)
    ifft(wenergy, tdenergy)
    cenergy = TimeSeries(tdenergy,
                         delta_t=tdenergy.delta_t, copy=False) # Normally delta_t is dur/tdenergy.size ... must figure out better way of doing this
    if normalized:
        energy = type(cenergy)(
            cenergy.real() ** 2. + cenergy.imag() ** 2.,
            delta_t=1, copy=False)
        medianenergy = energy.numpy().mean()
        result = energy / medianenergy
    else:
        result = cenergy
   
    return result

def padding(window_size, desired_size):
        """The `(left, right)` padding required for the IFFT
        :type: `tuple` of `int`
        """
        pad = desired_size - window_size
        return (int(pad/2.), int((pad + 1)/2.))

def get_data_indices(dur, f0, indices):
    """Returns the index array of interesting frequencies for this row
    """
    return np.round(indices + 1 +
                       f0 * dur).astype(int)

def _get_indices(windowsize):
    half = int((windowsize - 1) / 2.) 
    return np.arange(-half, half + 1)

def get_window(size, f0, qprime):
    """Generate the bi-square window for this row
    Returns
    -------
    window : `numpy.ndarray`
    """

    # dimensionless frequencies
    xfrequencies = np.linspace(-1., 1., size)

    # normalize and generate bi-square window
    norm = np.sqrt(315. * qprime / (128. * f0))
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
    ap.add_argument("-f", "--freq-res", required=False, type=float,
        default=0.1, help="frequency resolution")
    ap.add_argument("-t", "--t-res", required=False, type=float,
        default=0.001, help="time resolution")
    ap.add_argument("--f-start", required=False, type=float,
        help="plot start frequency")
    ap.add_argument("--f-end", required=False, type=float,
        help="plot end frequency")

    args = ap.parse_args()


    #Initialize parameters
    out_dir = args.output_dir
    now = args.usertag
    os.makedirs('%s/run_%s' % (out_dir,now))  # Fail early if the dir already exists
    normalized = args.normalize # Set this as needed
    sampling = args.samp_freq #sampling frequency
    mismatch=.2
    qrange=(4,64)
    frange=(0,np.inf)
    tres = args.t_res
    fres = args.freq_res

    # Read data and remove low frequency content
    fname = 'H-H1_LOSC_4_V2-1126259446-32.gwf'
    url = "https://losc.ligo.org/s/events/GW150914/" + fname
    urllib.urlretrieve(url, filename=fname)
    h1 = read_frame('H-H1_LOSC_4_V2-1126259446-32.gwf', 'H1:LOSC-STRAIN')
    psd = interpolate(welch(h1), 1.0 / 32)
    #h1 = TimeSeries(np.random.normal(size=64*4096), delta_t = 1. / sampling)
    h1 = highpass_fir(h1, 15, 8)
    white_strain = (h1.to_frequencyseries() / psd ** 0.5 * psd.delta_f).to_timeseries()
    h1 = white_strain

    #perform Q-tiling
    Qbase, frange = qtiling(h1, qrange, frange, sampling, normalized, mismatch)

    #Choose Q-plane and plot
    qplane, interp_func = Qplane(Qbase, h1, sampling, normalized, out_dir, now, frange, tres, fres)

    #Plot spectrogram
    if args.f_start:
        frange[0] = args.f_start
    if args.f_end:
        frange[1] = args.f_end
    plotter(interp_func, out_dir, now, frange, h1, sampling, tres, fres)

    print 'Done!'

if __name__ == '__main__':
    main()
