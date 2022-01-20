# -*- coding: utf-8 -*-
# @Time    : 2022-01-19 10:55 p.m.
# @Author  : young wang
# @FileName: spectro_view.py
# @Software: PyCharm

import numpy as np
from OssiviewBufferReader import OssiviewBufferReader
from matplotlib import pyplot as plt
from numpy.fft import fft

if __name__ == '__main__':
    file = '../data/2020-Sep-30  12.04.30 PM.bin'

    reader = OssiviewBufferReader(file)
    data = np.array(reader.data["DAQ Buffer"], dtype='float64')
    #
    div = reader.header.metaData['Header']['Meta Data']['DIV']
    div = np.array(div)
    div = div[div < data.shape[3]]

    clean = np.zeros([data.shape[0], data.shape[2], data.shape[3] - len(div)], dtype=data.dtype)

    for g in range(data.shape[0]):
        for i in range(0, data.shape[2]):
            k = 0
            for j in range(data.shape[3]):
                if j not in div:
                    clean[g, i, k] = data[g, :, i, j]
                    k = k + 1

    A_lines = fft(clean, axis=2).reshape(-1, 1460)

    plt.plot(20*np.log10(abs(A_lines[0,:])))
    plt.show()