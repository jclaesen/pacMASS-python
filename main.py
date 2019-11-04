#!/usr/bin/env python3

import pandas as pd
import os
import sys

sys.path.append("..")
from pacMASS import preprocess
from pacMASS import calculateAC
from pacMASS import writeOutputFile

###############################################################################

def init():
    _read_AC()
    _read_ReRa()

def _read_AC():
    global _AC
    #_AC = pd.read_csv("./data/AC_matrix_2.txt", sep="\t").values
    _AC = pd.read_csv(os.path.join(os.path.dirname(__file__),".\\data\\AC_matrix_2.txt"), sep="\t").values
                      
def getAC():
    global _AC
    return _AC

def _read_ReRa():
    global _ReRa
    #_ReRa = pd.read_csv("./data/RelRatio_matrix.txt", sep="\t").values
    _ReRa = pd.read_csv(os.path.join(os.path.dirname(__file__),".\\data\\RelRatio_matrix.txt"), sep="\t").values
def getReRa():
    global _ReRa
    return _ReRa


def pacmass (monoMassInput, numSList, filename='', ppm=10, alpha=0.05, columns=["m/z", "Charge"]):
    '''predicting the elemental composition of peptides and small proteins based on the monoisotopic mass
    
    Parameters
    ----------
    
    monoMassInput: float, list or string
        A single monoisotopic mass, list of monoisotopic masses, file containing monoisotopic masses 
    numSList: list
        The number of sulphur-atoms
    filename: string
        Name of the file (txt or csv) where results will be saved

    
    Returns
    -------
    
    results: numpy.ndarray or list of numpy.ndarray's
        Elemental compositions predicted with pacMASS based on the monoisotopic mass
        
        Column 0: number of Carbon-atoms
        Column 1: number of Hydrogen-atoms
        Column 2: number of Nitrogen-atoms
        Column 3: number of Oxygen-atoms
        Column 4: number of Sulphur-atoms
        Column 5: calculated monoisotopic mass (neutral)
        Column 6: monoMassInput        
    '''

    init()
       
    # Importing atom composition file
    ac = getAC()
    # Importing relative isotope intensities
    RR = getReRa()

    if isinstance(numSList, int):
        numSList = list(map(int, str(numSList)))

    monoMass = preprocess.handleInput(monoMassInput, columns)
    if monoMass is None:
        print("Error: The specified mass is not within the allowed mass boundaries.")
        return
    elif len(monoMass)==0:
        print("Error: The specified masses are not within the allowed mass boundaries.")
        return
    
    totalResults = []
        
    for n in monoMass:
        results = []
            
        for nS in numSList:
            result = calculateAC.calculateAC(n, RR, ac, nS, ppm, alpha)
                
            if result.size !=0:
                results.append(result)
            
        if len(results) != 0:
            totalResults.extend(results)
         
    if(len(filename)!=0):
        writeOutputFile.writeOutputFile(totalResults, filename)
        print("Results are written to file")
    else:
        return(totalResults)
    
 
if __name__ == "__main__":
    results = pacmass(monoMassInput=1045.4, numSList=[0], ppm=10, alpha=0.05, columns=["m/z", "Charge"])
    print("Result for a molecule with mass 1045.4 dalton, without S-atoms:", results)