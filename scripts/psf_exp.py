# -*- coding: utf-8 -*-
# @Time    : 2021-01-21 4:17 p.m.
# @Author  : young wang
# @FileName: psf_exp.py
# @Software: PyCharm


"""
1. https://www.sweptlaser.com/clean-optical-performance
2. https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-3-2632&id=279023
3. https://opg.optica.org/abstract.cfm?uri=acp-2011-831116
4. https://www.spiedigitallibrary.org/conference-proceedings-of-
spie/8213/82130T/Long-coherence-length-and-linear-sweep-without-an-external-
optical/10.1117/12.911477.short
This script is to calculate the PSF for the the thesis proposal
this is the learned psf, not the actual one

"""
import numpy as np
import os

from scipy.signal import chirp, find_peaks, peak_widths
from matplotlib import pyplot as plt
import pickle
import matplotlib

if __name__ == '__main__':
    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'stixgeneral',
            'mathtext.fontset': 'stix',
        }
    )
    # load measured point spread function

    PATH = '../data/psf'
    with open(PATH, 'rb') as f:
        psf = pickle.load(f)
        f.close()

    # define the conversion factor from pixel index to depth
    # (centre_wavelength)^2/(2*bandwidth)
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3519436/
    K = (1550e-9) ** 2 / (2 * 40e-9)

    # the resulting unit is set at mm
    x = np.linspace(0, len(psf), len(psf)) * K * 1000

    # converting into the log unit
    psf_dB = np.ravel(20 * np.log10(abs(psf)))
    psf_dB -= psf_dB.min()

    # use find_peaks to get the indices of the peaks

    indices, _ = find_peaks(psf_dB, height=np.mean(psf_dB))

    # use find_peaks to get the indices of the peaks
    # get the highest peak index and value
    highest_peak_index = indices[np.argmax(psf_dB[indices])]
    highest_peak_value = psf_dB[highest_peak_index]
    print("Highest peak is at index", highest_peak_index, "with value", highest_peak_value)

    # create a mask for peaks within x samples of the highest peak
    mask_size = 10
    mask = (indices >= highest_peak_index - mask_size) & (indices <= highest_peak_index + mask_size)

    # get the peaks within 20 samples of the highest peak
    nearby_peaks_indices = indices[mask]
    nearby_peaks_values = psf_dB[nearby_peaks_indices]

    # sort the nearby peaks by value, in descending order
    nearby_sorted_indices = nearby_peaks_indices[np.argsort(-nearby_peaks_values)]

    # find the second highest peak that is different from the highest peak
    for index in nearby_sorted_indices:
        if psf_dB[index] < highest_peak_value:
            second_highest_peak_index = index
            second_highest_peak_value = psf_dB[index]
            break
    print("Second highest peak within {} samples is at index "
          "{} with value {}".format(mask_size,
                                    second_highest_peak_index,
                                    second_highest_peak_value))

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.plot(x, psf_dB)

    x_indice, offset = second_highest_peak_index * K * 1000, 0.25
    x_left = x_indice - offset
    x_right = x_indice + offset
    #
    ax.hlines(y=psf_dB.max(), xmin=x_left, xmax=x_right, colors='r', linestyle='dotted', linewidth=2.5)
    ax.hlines(y=second_highest_peak_value, xmin=x_left, xmax=x_right, colors='r', linestyle='dotted', linewidth=2.5)

    ax.annotate("",
                xy=(x_indice, second_highest_peak_value), xycoords='data',
                xytext=(x_indice, psf_dB.max()), textcoords='data',
                arrowprops=dict(arrowstyle="<->",
                                connectionstyle="arc3", color='r', lw=2.5),
                )
    ax.text(x_left + 0.05, psf_dB.mean() + (highest_peak_value - second_highest_peak_value)/2,
            'PSF - ''sidelobe: %.2f dB' % (highest_peak_value - second_highest_peak_value),
            fontsize=15, fontweight='bold', rotation='vertical')

    # define the range to exclude
    include_range = 15

    # create a mask that excludes the range around the peak
    mask = np.zeros(psf_dB.shape, dtype=bool)
    mask[max(0, highest_peak_index - include_range):min(len(psf_dB), highest_peak_index + include_range + 1)] = True

    # set the peak itself to False in the mask
    mask[highest_peak_index] = False

    # create a new array excluding the range around the main peak
    arr_excluding_range = psf_dB[mask]

    # calculate the average of the new array
    avg_excluding_range = np.mean(arr_excluding_range)

    print(f"The average value excluding the main peak is {avg_excluding_range}")

    ax.hlines(y=psf_dB.max(), xmin=x_left + 2.6, xmax=x_right + 2.7, colors='r', linestyle='dotted', linewidth=2.5)
    ax.hlines(y=avg_excluding_range, xmin=x_left + 2.6, xmax=x_right + 2.7, colors='r', linestyle='dotted',
              linewidth=2.5)

    ax.annotate("",
                xy=(x_right + 2.5, avg_excluding_range), xycoords='data',
                xytext=(x_right + 2.5, psf_dB.max()), textcoords='data',
                arrowprops=dict(arrowstyle="<->",
                                connectionstyle="arc3", color='r', lw=2.5),
                )
    ax.text(x_right + 2.3, psf_dB.mean() + (highest_peak_value - avg_excluding_range)/3,
            'PSF - ''background: %.2f dB' % (highest_peak_value - avg_excluding_range),
            fontsize=15, fontweight='bold', rotation='vertical')

    ax.set_ylabel('amplitude [dB]', fontweight='bold')
    ax.set_xlabel('depth [mm]', fontweight='bold')
    ax.set_title('averaged axial point spread function (PSF)', fontweight='bold')

    ax.minorticks_on()
    ax.tick_params(which='minor', width=2)  # Set the width of the minor ticks to 1 (you can adjust this value)

    plt.tight_layout()
    # ax.tick_params(which='minor', width=4)  # Set the width of the minor ticks to 1 (you can adjust this value)

    # desktop_path = os.path.join(os.getenv('HOME'), 'Desktop')
    # file_path = os.path.join(desktop_path, 'Master/Thesis/Figure/Chapter 1/1.19 PSF setup/1.19 PSF result.svg')
    # plt.savefig(file_path, dpi=600,
    #             format='svg',
    #             bbox_inches='tight', pad_inches=0,
    #             facecolor='auto', edgecolor='auto')
    plt.show()
    plt.close(fig)