# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:39:15 2019

@author: Drew Hubley
"""

from OssiviewBufferReader import OssiviewBufferReader
import yaml
import numpy as np

def class Buffer:
    def __init__(self, commonName, bufferID, data):
        self.commonName = commonName
        self.bufferID = bufferID
        self.data = data
    
    def getData():
        return self.data

def class OssiviewBufferExporter:
    def __init__(self,exportPath):
        self._exportPath = exportPath
        self._data = [] #define an empty list of np arrays
        self._header = {} #define an empty header
        

    
    def _getType(self):
        return {'struct float2' : np.complex64,
                'struct DopplerData' : np.complex64,
                'unsigned short' : np.uint16,
                'float' : np.float32}
        
if __name__ == "__main__":
    exportOut = OssiviewBufferExporter("TestOut.bin")
    
    
        
        