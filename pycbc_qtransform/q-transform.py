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

from math import pi, ceil, log
import numpy as np
from pycbc.types.timeseries import FrequencySeries, TimeSeries
import sys
from pycbc.frame import read_frame
from pycbc.filter import highpass_fir, matched_filter
from pycbc.waveform import get_fd_waveform
from pycbc.psd import welch, interpolate
import urllib
import datetime
from scipy.signal import tukey
from numpy import fft as npfft

__author__ = 'Hunter Gabbard <hunter.gabbard@ligo.org>'
__credits__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

def qtransform(data, Q, f0, sampling, normalized):

    """
    Parameters
    ----------
    data : `LIGO gwf frame file`
        raw time-series data set
    normalized : `bool`, optional
        normalize the energy of the output, if `False` the output
        is the complex `~numpy.fft.ifft` output of the Q-tranform
    """

    #Q-transform data for each (Q, frequency) tile

    #Initialize parameters
    qprime = Q / 11**(1/2.) # ... self.qprime
    dur = int(len(data)) / sampling # length of your data chunk in seconds ... self.duration
    fseries = TimeSeries.to_frequencyseries(data)
      
    #Window fft
    window_size = 2 * int(f0 / qprime * dur) + 1
        
    #Get indices
    indices = _get_indices(dur)

    #Apply window to fft
    windowed = fseries[get_data_indices(dur, f0, indices)] * get_window(dur, indices, f0, qprime, Q, sampling)

    # pad data, move negative frequencies to the end, and IFFT
    padded = np.pad(windowed, padding(window_size, dur, f0, Q), mode='constant')
    wenergy = npfft.ifftshift(padded)

    # return a `TimeSeries`
    tdenergy = npfft.ifft(wenergy)
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
    #Initialize parameters
    normalized = True # Set this as needed
    Q = 20 # self-explanatory ... self.q
    f0 = 5 # initiail frequency ... self.frequency
    sampling = 4096 #sampling frequency

    # Read data and remove low frequency content
    fname = 'H-H1_LOSC_4_V2-1126259446-32.gwf'
    url = "https://losc.ligo.org/s/events/GW150914/" + fname
    urllib.urlretrieve(url, filename=fname)
    h1 = read_frame('H-H1_LOSC_4_V2-1126259446-32.gwf', 'H1:LOSC-STRAIN')
    h1 = np.random.normal(size=256*4096)
    h1 = highpass_fir(h1, 15, 8)

    # Calculate the noise spectrum
    psd = interpolate(welch(h1), 1.0 / 32)

    #perform q-transform on timeseries
    q = qtransform(h1, Q, f0, sampling, normalized)

if __name__ == '__main__':
    main()
