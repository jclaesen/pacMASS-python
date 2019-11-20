""" preprocess.py
    This module implements the functions to handle the different sources of input for the pacMASS package
    It also converts the masses to their neutral mass
"""

import pandas as pd
import numpy as np
import os
import sys

def calculateMonoMass(inputDF):
    """
    Function that calculates the neutral monoisotopic mass
    
    Parameters
    ----------
        inputDF: numpy array
            measured masses and charges
        
    Returns
    -------
        monoMass: list
            Neutral monoisotopic masses
    """
    monoMass = list()

    for row in inputDF.iterrows():   # row[1][0]=charge, row[1][1]=m/z
        monoMass.append(row[1][1] * row[1][0] - row[1][0] * 1.009794)

    return monoMass

def filterMonoMass(monoMass, lowerLimit, upperLimit):
    """
    Function that filters (array of) monoisotopic mass

    Parameters
    ----------
    
        monoMass: float or numpy array
        lowerLimit: float
        upperLimit: float
        
    Returns
    -------
        monoMassFiltered: list 
            Filtered monoisotopic masses
    """
    
    if isinstance(monoMass, float):
        if((monoMass >= lowerLimit) & (monoMass <= upperLimit)):
            return([monoMass])
        else:
            sys.exit("The specified monoisotopic mass is not within the allowed mass boundaries")
            
    elif isinstance(monoMass, np.ndarray):
        if monoMass.dtype=='float64':

            down = monoMass >= lowerLimit
            up = monoMass <= upperLimit
        
            index = np.where(down & up)
            monoMassFiltered = list(monoMass[index])

            return monoMassFiltered
        else:
            sys.exit("The specified monoisotopic masses are not defined as float")
def handleInput(monoMassInput, columns):
    """
    Parameters
    ----------
    
        monoMassInput: float, list or string
            A single monoisotopic mass (neutral), list of monoisotopic masses (neutral), file containing measured masses (with charge) 
    
    Returns
    -------
    
        monoMassOut: list
            Neutral monoisotopic mass(es)

    """
    
    if not isinstance(monoMassInput, str) and not isinstance(monoMassInput, float) and not isinstance(monoMassInput, list):
        sys.exit("Argument 'monoMassInput' should be a list, float or a string")
  
    if not isinstance(columns, list) or not len(columns)==2:
        sys.exit("Argument 'columns' should be a list of length 2")

    
    if isinstance(monoMassInput, str):
        if os.path.isfile(monoMassInput):
            print("importing mass input file...")

            if monoMassInput.endswith(".txt"):
                try:
                    mz = pd.read_csv(monoMassInput, delimiter="\t", usecols=[columns[0], columns[1]], dtype={columns[0]: float, columns[1]: float})
                except Exception:
                    sys.exit("Given column names don't match file")
        
            elif monoMassInput.endswith(".csv"):
                try:
                    mz = pd.read_csv(monoMassInput, delimiter=",", usecols=[columns[0], columns[1]], dtype={columns[0]: float, columns[1]: float})
                except Exception:
                    sys.exit("Given column names don't match file")
        
            else: 
                sys.exit("Error: File can not be opened : \"{}\"".format(monoMassInput))
            
            
        print("calculating monoisotopic mass...")
        monoMassOut = calculateMonoMass(mz)

    if isinstance(monoMassInput, float):

        monoMassOut = filterMonoMass(monoMassInput, 0, 4000)

                
    elif isinstance(monoMassInput, list):

        monoMassArray = np.array(monoMassInput)
        monoMassOut = filterMonoMass(monoMassArray, 0, 4000)
        
    return(monoMassOut)
