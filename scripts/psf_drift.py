# -*- coding: utf-8 -*-
# @Time    : 2021-02-08 8:57 p.m.
# @Author  : young wang
# @FileName: psf_drift.py
# @Software: PyCharm

from pathlib import Path
import numpy as np
import pickle
from matplotlib import pyplot as plt
# from sporco import metric
# from sporco import cnvrep
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

matplotlib.rcParams.update({'font.size': 18})
path = Path('../data/2021.03.02 Measurement/processed')

files = []

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

for x in path.iterdir():
    files.append(x)

files.sort()
files.pop(0)
psf = np.zeros((330, 1, len(files)), complex)
for i in range(len(files)):
    with open(files[i], 'rb') as f:
        psf[:, :, i] = pickle.load(f)
        f.close()

# phase = np.zeros((330,len(files)))
# fig,ax = plt.subplots(1,2,figsize=(16,9))
# mse = np.zeros(len(files))
# for i in range(psf.shape[2]):
#
#     temp = abs(psf[:,:,i])
#     ax[0].plot(np.roll(temp,int(40*i-150)),label=str(i*10) +' sec')
#     ax[0].legend()
#     mse[i] = metric.mse(abs(psf[:,:,0]),abs(psf[:,:,i]))
#     phase[:,i] = np.angle(psf[:,:,i].squeeze())
#
# mse = 100*mse.mean()
#
# ax[0].set_xlabel('axial depth: pixel')
# ax[0].set_ylabel('normalized amplitude')
# ax[0].set_title('axial PSF magnitude plot: average mse value: %.2f %%' %mse)
#
# ax[1].plot(phase)
# ax[1].set_ylabel('rad')
# ax[1].set_title('axial PSF phase plot')
# ax[1].set_xlabel('axial depth: pixel')
# plt.tight_layout()
# plt.show()

phase_time = np.zeros(len(files))
mag_time = np.zeros(len(files))
fig = plt.figure(constrained_layout=True,figsize=(18,13))
gs = fig.add_gridspec(ncols=3, nrows=4)

fig.suptitle('axial PSF over 60 s')

mse = np.zeros(len(files))
time = np.linspace(0, 60, len(files))
cross = np.zeros(len(files))

for i in range(psf.shape[2]):
    temp = psf[:, :, i]
    mag_time[i] = abs(temp).max()

    mag_time[i] = abs(temp).max()
    a = abs(psf[:, :, 0]).squeeze()
    b = abs(temp).squeeze()

    cross[i] = np.correlate(a, b, mode='same').max()

    phase = np.angle(temp, deg=False)
    phase_time[i] = phase[np.argmax(abs(temp))]

ax = fig.add_subplot(gs[0,0])
ax.stem(time, mag_time)
ax.set_xlabel('time: s')
ax.set_ylabel('magnitude(a.u.)')
ax.set_title('axial PSF peak magnitude')

ax = fig.add_subplot(gs[0,1])
ax.stem(time, cross)
ax.set_title('mag xcorr average: %.4f' % cross.mean())
ax.set_xlabel('time: s')

# unwrap phase along the time
ax = fig.add_subplot(gs[0,2])
ax.stem(time, np.unwrap(phase_time))
ax.set_ylabel('rad')
ax.set_title('axial PSF peak phase')
ax.set_xlabel('time: s')

# psf drift over the first frame: first 15,000 line = 3/20 seconds
path = '../data/2021.03.02 Measurement/frame/12.06.59'
with open(path, 'rb') as f:
    frame = pickle.load(f)
    f.close()
from numpy import linalg as LA
#
frame = normalise(frame, dimN=1)
# frame = LA.norm(frame, ord= None, axis = -1)

#
# sample 100 lines: every 600 lines out of the 15,000 lines: spaced out in 3/500 second
sample = 100

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
for i in range(frame_sample.shape[1]):
    temp = frame_sample[:, i]
    peak_mag[i] = abs(temp).max()

    phase = np.angle(temp, deg=False)
    peak_phase[i] = phase[np.argmax(abs(temp))]

    a = abs(frame_sample[:, 0]).squeeze()
    b = abs(temp).squeeze()

    peak_cross[i] = np.correlate(a, b, mode='same').max()

ax = fig.add_subplot(gs[1,0])
ax.stem(time, peak_mag)
ax.set_xlabel('time: ms')
ax.set_ylabel('magnitude(a.u.)')
ax.set_title('axial PSF peak magnitude')

ax = fig.add_subplot(gs[1,1])
ax.stem(time, peak_cross)
ax.set_ylim(0.90,1)
ax.axhline(y = peak_cross.mean(),color='red',linestyle='--' )
ax.set_title('mag xcorr average: %.4f' % peak_cross.mean())
ax.set_xlabel('time: ms')

ax = fig.add_subplot(gs[1,2])
ax.stem(time, np.unwrap(peak_phase))
ax.set_xlabel('time: ms')
ax.set_ylabel('rad')
ax.set_title('axial PSF peak phase')

fs = 100e3
# number of lines
nol = 2000
frame_half = frame[:, 0:nol]
peak_mag = np.zeros(frame_half.shape[1])
peak_phase = np.zeros(frame_half.shape[1])
peak_cross = np.zeros(frame_half.shape[1])

for i in range(frame_half.shape[1]):
    temp = frame_half[:, i]
    peak_mag[i] = abs(temp).max()

    phase = np.angle(temp, deg=False)
    peak_phase[i] = phase[np.argmax(abs(temp))]

    a = abs(frame_half[:, 0]).squeeze()
    b = abs(temp).squeeze()

    peak_cross[i] = np.correlate(a, b, mode='same').max()
#
duration = int(frame_half.shape[1] * 1000 / fs)
time = np.linspace(0, duration, num=frame_half.shape[1])

ax = fig.add_subplot(gs[2,0])
ax.stem(time, peak_mag)
ax.set_xlabel('time: ms')
ax.set_ylabel('magnitude(a.u.)')
ax.set_title('axial PSF peak magnitude')

ax = fig.add_subplot(gs[2,1])
ax.stem(time, peak_cross)
ax.set_ylim(0.95,1)
ax.axhline(y = peak_cross.mean(),color='red',linestyle='--' )
ax.set_title('mag xcorr average: %.4f' % peak_cross.mean())
ax.set_xlabel('time: ms')

ax = fig.add_subplot(gs[2,2])
ax.stem(time, np.unwrap(peak_phase))
ax.set_xlabel('time: ms')
ax.set_ylabel('rad')
ax.set_title('axial PSF peak phase')

ax = fig.add_subplot(gs[3,:])
# axins = zoomed_inset_axes(ax, zoom=0.6, loc=2)
# axins.set_xticks([])
# axins.set_yticks([])

ax.set_title('sample axial PSF magnitude plot')
for i in range(4):
    temp = 20 * np.log10(abs(psf[:, :, 15 * i]))
    temp += 75
    # temp = (abs(psf[:, :, 15 * i]))
    ax.plot(temp,label=str(i*15) +' sec')
    ax.legend(loc = 'upper right',fontsize=(8))
    ax.set_xlabel('axial depth: pixel')
    ax.set_ylabel('intensity (dB)')

    # axins.plot(temp[0:50])
    # axins2 = zoomed_inset_axes(ax, zoom=0.5, loc=2)
    # zo_temp = temp
    # axins2.plot(zo_temp)
    # axins2.set_xlim(0, 50)
    # axins2.set_xticks([])
    # axins2.set_yticks([])

    ax.set_aspect(30/25)

plt.show()


fig, ax = plt.subplots(1,2, constrained_layout=True,figsize=(16,9))
ax[0].stem(time, peak_cross)
ax[0].set_ylim(0.95,1)
ax[0].axhline(y = peak_cross.mean(),color='red',linestyle='--' )
ax[0].set_title('mag xcorr average: %.4f' % peak_cross.mean(),fontname ='Arial')
ax[0].set_xlabel('time: ms',fontname ='Arial')

axins = ax[1].inset_axes([0.05, 0.6, 0.37, 0.37])
axins.set_xticks([])
axins.set_yticks([])

for i in range(5):
    # temp = 20*np.log10(abs(frame_half[:, i*600]))
    temp = np.abs(frame_half[:, i*499])

    # ax[1].plot(np.roll(temp,i*3), label=str(i * 5) + ' ms')
    ax[1].plot(temp, label=str(i * 5) + ' ms')

    ax[1].legend(loc='upper right', fontsize=(12))
    # temp += 60

    # zo_temp = temp[130:160]
    axins.plot(np.roll(temp,i*3))
    axins.plot(temp)

    # axins.plot(temp)

    axins.set_xlim(145, 163)
    axins.set_ylim(0, 0.05)
    # axins2.set_xticks([])
    # axins2.set_yticks([])

ax[1].indicate_inset_zoom(axins)
ax[1].set_xlabel('axial depth: pixel',fontname ='Arial')
ax[1].set_ylabel('amplitude: a.u.',fontname ='Arial')
ax[1].set_title('axial PSF magnitude',fontname ='Arial')


plt.show()

