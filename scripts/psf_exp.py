# -*- coding: utf-8 -*-
# @Time    : 2021-01-21 4:17 p.m.
# @Author  : young wang
# @FileName: psf_exp.py
# @Software: PyCharm


"""
This script is to calculate the PSF for the the thesis proposal
this is the learned psf, not the actual one

"""
import numpy as np
from scipy.signal import chirp, find_peaks, peak_widths
from matplotlib import pyplot as plt
import pickle
import matplotlib

matplotlib.rcParams.update({'font.size': 20})

# load measured point spread function

PATH = '../data/psf'
with open(PATH,'rb') as f:
    psf = pickle.load(f)
    f.close()

# define the conversion factor from pixel index to depth
#(centre_wavelength)^2/(2*bandwidth)
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3519436/
K = (1550e-9)**2/(2*40e-9)

#the resulting unit is set at mm
x = np.linspace(0,len(psf),len(psf))*K*1000

# converting into the log unit
psf_dB = 20*np.log10(abs(psf))

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)

psf_dB -= psf_dB.min()
plt.plot(x,psf_dB)
plt.hlines(y = psf_dB.max(),xmin = 4, xmax = 6, colors= 'r', linestyle = 'dotted')
plt.hlines(y = psf_dB.mean(),xmin = 4, xmax = 6, colors= 'r', linestyle = 'dotted')

ax.annotate("",
            xy=(4.5, psf_dB.mean()), xycoords='data',
            xytext=(4.5, psf_dB.max()), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3", color='r', lw=2),
            )

plt.text(1,psf_dB.mean()+20, 'Dynamic range: %.2f dB' %(psf_dB.max() - psf_dB.mean()), fontsize = 20)
# rotation = 90, fontsize = 18)

plt.ylabel('amplitude [dB]')
plt.xlabel('depth [mm]')
# plt.minorticks_on()
# plt.xticks()
plt.title('Averaged axial point spread function (PSF)')
# plt.grid(b= True, which='minor', color='r', linestyle='--')
plt.tight_layout()
plt.show()
