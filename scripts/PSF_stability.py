# -*- coding: utf-8 -*-
# @Time    : 2021-02-25 5:16 p.m.
# @Author  : young wang
# @FileName: PSF_stability.py
# @Software: PyCharm

import pickle
# from sporco import cnvrep
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.gridspec as gridspec


def normalise(v, dimN=2):
    r"""Normalise vector components of input array.

    Normalise vectors, corresponding to slices along specified number
    of initial spatial dimensions of an array, to have unit
    :math:`\ell_2` norm. The remaining axes enumerate the distinct
    vectors to be normalised.

    Parameters
    ----------
    v : array_like
      Array with components to be normalised
    dimN : int, optional (default 2)
      Number of initial dimensions over which norm should be computed

    Returns
    -------
    vnrm : ndarray
      Normalised array
    """

    axisN = tuple(range(0, dimN))
    if np.isrealobj(v):
        vn = np.sqrt(np.sum(v**2, axisN, keepdims=True))
    else:
        vn = np.sqrt(np.sum(np.abs(v)**2, axisN, keepdims=True))
    vn[vn == 0] = 1.0
    return np.asarray(v / vn, dtype=v.dtype)


np.seterr(divide='ignore', invalid='ignore')
# Customize matplotlib
matplotlib.rcParams.update(
    {
        'font.size': 16,
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

fig,ax = plt.subplots(1,3,figsize=(16,9))

# psf drift over the first frame: first 15,000 line = 3/20 seconds
path = '../data/2021.03.02 Measurement/frame/12.06.59'
with open(path, 'rb') as f:
    frame = pickle.load(f)
    f.close()
# frame = cnvrep.normalise(frame, dimN=1)
frame = normalise(frame, dimN=1)

# sample 100 lines: every 600 lines out of the 15,000 lines: spaced out in 3/500 second
# sample = 100
sample = 15000

frame_sample = []
time = np.linspace(0, 3 / 20, sample) * 1000
n = int(frame.shape[1] / sample)
for i in range(frame.shape[1]):

    if i % n == 0:
        frame_sample.append(frame[:, i])

frame_sample = np.asarray(frame_sample).T
#
peak_mag = np.zeros(frame_sample.shape[1])
peak_phase = np.zeros(frame_sample.shape[1])
peak_cross = np.zeros(frame_sample.shape[1])

peak_comlex = np.zeros(frame_sample.shape[1],complex)
for i in range(frame_sample.shape[1]):
    temp = frame_sample[:, i]
    peak_mag[i] = abs(temp).max()

    loc = np.argmax(abs(temp))
    peak_comlex[i] = frame_sample[loc, i]

    phase = np.angle(temp, deg=False)
    peak_phase[i] = phase[np.argmax(abs(temp))]

    a = abs(frame_sample[:, 0]).squeeze()
    b = abs(temp).squeeze()

    peak_cross[i] = np.correlate(a, b, mode='same').max()

ax[0].stem(time, peak_mag)
ax[0].set_xlabel('time: ms')
ax[0].set_ylabel('magnitude')
ax[0].set_title('axial PSF peak magnitude')

ax[1].stem(time, peak_cross)
ax[1].set_title('mag xcorr average: %.4f' % peak_cross.mean())
ax[1].set_xlabel('time: ms')

ax[2].stem(time, np.unwrap(peak_phase))
ax[2].set_xlabel('time: ms')
ax[2].set_ylabel('rad')
ax[2].set_title('axial PSF peak phase')

plt.tight_layout()
plt.show()

dif_amp = []
dif_phase = []
phase = np.unwrap(peak_phase)
dif_complex = []
for i in range(len(peak_mag)-1):
    temp_amp = peak_mag[i+1] - peak_mag[i]
    temp_phase = (phase[i+1]) - (phase[i])
    temp_complex = peak_comlex[i+1] - peak_comlex[i]

    dif_complex.append(temp_complex)
    dif_amp.append(temp_amp)
    dif_phase.append(temp_phase)

dif_complex = np.asarray(dif_complex)
fig,ax = plt.subplots(2,3,figsize=(16,9))
ax[0,0].plot(peak_mag, label='magnitude')
ax[0,0].plot(dif_amp,color = 'red',label='magnitude difference')
# ax[0,0].set_ylim(-0.1,0.1)
ax[0,0].set_ylabel('magnitude(a.u.)')

secax = ax[0,0].secondary_yaxis('right',color = 'red')
secax.set_ylabel('magnitude difference(a.u.)')
# secax.plot(dif_amp,color = 'red',label='amplitude difference')

secax.set_ylim(-0.1,0.1)
ax[0,0].set_xlabel('measurements')

ax[0,0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.,fontsize = 12)

ax[0,1].plot(phase,label='phase')
ax[0,1].plot(dif_phase,color = 'red',label='phase difference')
ax[0,1].set_ylabel('phase(rad)')
ax[0,1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.,fontsize = 12)

secax = ax[0,1].secondary_yaxis('right', color = 'red')
secax.set_ylabel('phase difference(rad)')
# secax.set_ylim(2*np.min(dif_phase),2*np.max(dif_phase))
# secax.plot(dif_phase,color = 'red',label='phase difference')

ax[0,1].set_xlabel('measurements')

ax[0,2].scatter(peak_comlex.real,peak_comlex.imag,label = 'complex')
ax[0,2].scatter(dif_complex.real,dif_complex.imag,label = 'complex difference',color = 'red')
ax[0,2].set_xlabel('real part')
ax[0,2].set_ylabel('imaginary part')
ax[0,2].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.,fontsize = 12)

_, bins, _ = ax[1,0].hist(dif_amp, 1000, density=1)

mu_amp, sigma_amp = norm.fit(dif_amp)
best_fit_line = norm.pdf(bins, mu_amp, sigma_amp)
ax[1,0].plot(bins, best_fit_line)
ax[1,0].set_xlim(left=-0.1, right = 0.1 )

ax[1,0].set_xlabel('magnitude difference(a.u.)')
ax[1,0].set_ylabel('number of events')

textstr = r'$\sigma = %1.4f$ a.u.' %(sigma_amp)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)

ax[1,0].text(0.05, 0.98, textstr, transform=ax[1,0].transAxes, fontsize=15,
                  verticalalignment='top', bbox=props)

_, bins, _ = ax[1,1].hist(dif_phase, 1000, density=1)

mu_phase, sigma_phase = norm.fit(dif_phase)
best_fit_line = norm.pdf(bins, mu_phase, sigma_phase)
ax[1,1].plot(bins, best_fit_line)

ax[1,1].set_xlabel('phase difference(rad)')
ax[1,1].set_ylabel('number of events')
ax[1,1].set_xlim(left=-0.1, right = 0.1 )
textstr = r'$\sigma = %1.2f $ mrad' %(sigma_phase*1000)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)

ax[1,1].text(0.05, 0.98, textstr, transform=ax[1,1].transAxes, fontsize=15,
                  verticalalignment='top', bbox=props)

divider = make_axes_locatable(ax[1,2])
cax = divider.append_axes('right', size='5%', pad=0.10)

im = ax[1,2].hist2d(dif_complex.real,dif_complex.imag,bins = 100, density=1, cmap=plt.cm.jet)
ax[1,2].set_title('complex difference')
ax[1,2].set_xlabel('real part')
ax[1,2].set_ylabel('imaginary part')
fig.colorbar(im[3],cax = cax, orientation='vertical',label="number of events")

plt.tight_layout()
plt.show()

# displacement calculation
move = phase.max() - phase.min()
D = move*(1550e-9)/(4*np.pi)
D /= 1e-6
print('path length changed %.6f um'%D)


