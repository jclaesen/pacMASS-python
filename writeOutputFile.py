#!/usr/bin/env python3

import pandas as pd

def writeOutputFile(results, filename):
    """
    Parameters
    ----------
    
        results: list or list of numpy arrays
            predicted elemental compositions 
    
        filename: string

    """
    
    if filename.lower().endswith(".csv"):
        formatFile = "csv"

    elif filename.lower().endswith(".txt"):
        formatFile = "txt"

    else:
       print("Unsupported file format: " + filename)
       return
       
       
    output = pd.DataFrame(results[0], columns=["C", "H", "N", "O", "S", "calc_mass","input_neutral_mass"])
    for n in range(1,len(results)):
        if len(results[n]) != 0:
            output = output.append(pd.DataFrame(results[n], columns=["C", "H", "N", "O", "S", "calc_mass", "input_neutral_mass"]), ignore_index= True)          
    
    file = open(filename,"w",newline='')
    if formatFile == ".txt": 
        output.to_csv(file, sep="\t", index=False)

    elif formatFile == ".csv":
        output.to_csv(file, sep=",", index=False, line_terminator="")
    
    file.close()