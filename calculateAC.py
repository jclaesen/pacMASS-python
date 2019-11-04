import numpy as np
import pandas as pd

def calculateAC(totalWeight, RR, ac, numS, ppm = 10, alpha = 0.05):
    '''predict the atomic composition based on the monoisotopic mass
    
    Parameters
    ----------
    
    totalWeight: float
        single monoisotopic mass 
    numS: float
        The number of sulphur-atoms the elemental composition should have
    ppm: float 
        mass tolerance
    alpha: float
        significance level of the prediction intervals. Currenlty only 0.05 and 0.01 are allowed

    
    Returns
    -------
    
    results: numpy.ndarray
        Elemental compositions predicted based on the monoisotopic mass
        
        Column 0: number of Carbon-atoms
        Column 1: number of Hydrogen-atoms
        Column 2: number of Nitrogen-atoms
        Column 3: number of Oxygen-atoms
        Column 4: number of Sulphur-atoms
        Column 5: calculated monoisotopic mass (neutral)
        Column 6: monoMassInput        
    '''   

    weight = np.array([12, 1.0078250321, 14.0030740052, 15.9949146, 31.97207070])   # weight [C, H, N, O, S]
    tolerance = ppm * totalWeight / 10**6

    AC = ac[ac[:,4]==numS]
    RR2= RR[ac[:,4]==numS]
   
    # calculate prediction interval for isoRatios
    estimateRR = calculateIsoRatio(numS, totalWeight, alpha)     # columns: fit, lwb, upb 
    
    # filter based on isoRatio prediction interval
    #try in pandas
    ac2 = AC[(RR2[:,0] >= estimateRR[0,1]) & (RR2[:,0] <= estimateRR[0,2]) & (RR2[:,1] >= estimateRR[1,1]) & (RR2[:,1] <= estimateRR[1,2]) & (RR2[:,2] >= estimateRR[2,1]) & (RR2[:,2] <= estimateRR[2,2]) & (RR2[:,3] >= estimateRR[3,1]) & (RR2[:,3] <= estimateRR[3,2])]

    if ac2.size == 0:
       return np.array([])

    ## STEP 2 ## generating theoretical posiible numbers for C, H, N, O
    
    # calculating min and max from atom compositions
    minAC = np.amin(ac2, axis=0)    # min [C, H, N, O, S]
    maxAC = np.amax(ac2, axis=0)    # max [C, H, N, O, S]

    # calculating nominalMass
    nominalMass = calculateNomMass(numS, totalWeight, alpha)
    nominalMass = [round(x) for x in nominalMass]

    # N and H rule
    if nominalMass[1] == nominalMass[2]:
        ## N rule ##  if nomMass is even, num N is also even
        if minAC[2] % 2 == 0 and nominalMass[0] % 2 != 0:     
                minAC[2] -= 1                                     
            
        if minAC[2] % 2 != 0 and nominalMass[0] % 2 == 0:
            minAC[2] -= 1

        ## H rule ## if nomMass is divisible by 4, num of H is also divisible by 4
                  ## if nomMass is even, num of H is also even 
        remainderH = minAC[1] % 4
        remainderNomMass = nominalMass[0] % 4

        if remainderNomMass == 0 and remainderH !=0:
            minAC[1] =  minAC[1] - remainderH
        
        elif remainderNomMass == 1:    # remainderH has to be 3
            if remainderH == 0:
                minAC[1] =  minAC[1] - 1
            elif remainderH == 1:
                minAC[1] =  minAC[1] - 2 
            elif remainderH == 2:
                minAC[1] =  minAC[1] - 3
        
        elif remainderNomMass == 2:
            if remainderH == 0:
                minAC[1] =  minAC[1] - 2
            elif remainderH == 1:
                minAC[1] =  minAC[1] - 3 
            elif remainderH == 3:
                minAC[1] =  minAC[1] - 1

        elif remainderNomMass == 3:    # remainderH has to be 1
            if remainderH == 0:
                minAC[1] =  minAC[1] - 3
            elif remainderH == 2:
                minAC[1] =  minAC[1] - 1
            elif remainderH == 3:
                minAC[1] =  minAC[1] - 2

    rangeAC = np.array([                            # rows = [C, H, N, O, S]
        np.arange(0,maxAC[0]-minAC[0] + 1, 1),
        np.arange(0,maxAC[1]-minAC[1] + 1, 4),
        np.arange(0,maxAC[2]-minAC[2] + 1, 2),
        np.arange(0,maxAC[3]-minAC[3] + 1, 1),
        np.arange(0,maxAC[4]-minAC[4] + 1, 1)])   

    ## STEP 3 ## Generating all combinations and mass based filter

    mass = np.array([                           # rows = [C, H, N, O, S]
        list(x * weight[0] for x in rangeAC[0]),
        list(x * weight[1] for x in rangeAC[1]),
        list(x * weight[2] for x in rangeAC[2]),
        list(x * weight[3] for x in rangeAC[3]),
        list(x * weight[4] for x in rangeAC[4])
    ])

    # create combination arrays
    nbComb = len(rangeAC[0])*len(rangeAC[1])*len(rangeAC[2])*len(rangeAC[3])*len(rangeAC[4])
    combinationAC = np.array(np.meshgrid(rangeAC[0], rangeAC[1], rangeAC[2], rangeAC[3], rangeAC[4])).reshape(5, nbComb).T
    combinationMass = np.array(np.meshgrid(mass[0], mass[1], mass[2], mass[3], mass[4])).reshape(5, nbComb).T

    totalMasses = combinationMass.sum(axis = 1)

    func = lambda x: x + np.sum(minAC * weight)
    totalMass = func(totalMasses)

    down = totalMass <= (totalWeight + tolerance)
    up = totalMass >= (totalWeight - tolerance)

    indexMass = np.where(down & up)

    totalMass = totalMass[indexMass]
    comboAC = combinationAC[indexMass]

    # creating results array
    func = lambda x: x + minAC
    results = func(comboAC[:,0:5])

    results = np.concatenate((results, totalMass[:, None]), axis=1)

    moleculeFill = np.full((results.shape[0],1), fill_value=totalWeight, dtype="float64") 
    results = np.concatenate((results, moleculeFill), axis=1)

    ## STEP 4 ## applying Senior's theorem + creating result array 

    if results.shape[1] != 0:
        if nominalMass[1] != nominalMass[2]: #if step 2 is not possible
            valences = np.array([4, 1, 5, 6, 6, 0])

            func = lambda x: x * valences
            results[:,0:6] = func(results[:,0:6])

            valenceMultiplication = results[:,0:6]
            valanceCondition = valenceMultiplication.sum(axis = 1) % 2 == 0
            results = results[valanceCondition]
            return results
        else:
            return results
    else: print("something went wrong")
    

def calculateNomMass(numS, monoMass, alpha):
    '''calculate the nominal mass of a peptide or protein based on the monoisotopic mass and the number of S-atoms
    
    Parameters
    ----------
    
    numS: float
        the number of sulphur-atoms
    monoMass: float 
        the neutral monoisotopic mass
    alpha: float
        significance level of the prediction interval. Currenlty only 0.05 and 0.01 are allowed

    
    Returns
    -------
    
    results: numpy.ndarray
        Prediction interval of the nominal mass based on the monoisotopic mass
        
        Column 0: Relative ratio's
        Column 1: lower bound of the prediction interval
        Column 2: upper bound of the prediction interval
    
    '''   

    if numS == 0:
        beta = [-0.0281454811,  0.9995094428]
        MvNM = 0.00339356789
        corrFact  = 1.00000380
        meanMass = 1197.76598
        diffMass = 113449218239

    elif numS == 1:
        beta = [0.00962820343, 0.99950448286]
        MvNM = 0.00441192839
        corrFact  = 1.00000821
        meanMass = 1524.47582
        diffMass = 78595101626

    elif numS == 2:
        beta = [0.0551256003, 0.9994982808]
        MvNM = 0.00599835281
        corrFact  = 1.00002548
        meanMass = 1948.05935
        diffMass = 29653059529

    elif numS == 3:
        beta = [0.104524965, 0.999492285]
        MvNM = 0.00812510070
        corrFact  = 1.00008876
        meanMass = 2376.95397
        diffMass = 7870618506

    elif numS == 4:
        beta = [0.147943249, 0.999492098]
        MvNM = 0.01110538580
        corrFact  = 1.00031260
        meanMass = 2708.08370
        diffMass = 1808231040

    elif numS == 5:
        beta = [0.209461865, 0.999489582]
        MvNM = 0.01232500256
        corrFact  = 1.00098912
        meanMass = 2934.94833
        diffMass = 451568554

    elif numS == 6:
        beta = [0.175224999, 0.999522901]
        MvNM = 0.01537179455
        corrFact  = 1.00289855
        meanMass = 3101.84379
        diffMass = 162241578

    else:
        beta = [0.188238156, 0.999551461]
        MvNM = 0.01871637
        corrFact  = 1.00671141
        meanMass = 3201.99966
        diffMass = 58949427
        
    nomMass = beta[0] + beta[1]*monoMass

    if alpha == 0.05:
        k = 1.96
    elif alpha == 0.01:
        k = 3.29
    else:
        print("alpha = {} cannot be used, set alpha to 0.05".format(alpha))
        k = 1.96

    lwb = nomMass - k * (MvNM)**0.5 * (corrFact + (monoMass - meanMass)**2 / diffMass)**0.5
    upb = nomMass + k * (MvNM)**0.5 * (corrFact + (monoMass - meanMass)**2 / diffMass)**0.5

    return np.array([nomMass, lwb, upb])

def calculateIsoRatio(numS, monoMass, alpha):

    '''estimate the prediction interval for the relative ratios R1, R2, R3, R4
    
    Parameters
    ----------
    
    numS: float
        the number of sulphur-atoms
    monoMass: float 
        the neutral monoisotopic mass
    alpha: float
        significance level of the prediction interval. Currenlty only 0.05 and 0.01 are allowed

    
    Returns
    -------
    
    output: pandas.DataFrame
        Prediction interval of the nominal mass based on the monoisotopic mass
        
        Column 0: nominal mass
        Column 1: lower bound of the prediction interval
        Column 2: upper bound of the prediction interval
    
    '''   

    if numS == 0:
        beta0 = np.array([-0.0189816820,  0.060423108,  0.0305081511,  0.0301270267])
        beta1 = np.array([0.5674546622,  0.235662205,  0.2217594086,  0.1735174427])
        beta2 = np.array([-0.0216932234,  0.029637619, -0.0238256966, -0.0174466220])
        beta3 = np.array([0.0075673616, -0.009869113,  0.0065015479,  0.0039611966])
        beta4 = np.array([-0.0009059258,  0.001128834, -0.0006612119, -0.0003479714])
        
        MvRR  = np.array([0.001161310, 0.0001495041, 3.082393e-05, 2.037520e-05])
        corrFact  = 1.000004
        meanMass  = 1.197766
        diffMass  = 113449.21824

    elif numS == 1:
        beta0 = np.array([-0.025138947,  0.42040388,  0.053970091,  0.0847557940])
        beta1 = np.array([0.565173102, -0.29035784,  0.321820468,  0.1669168496])
        beta2 = np.array([-0.022024778,  0.35711591, -0.115852624, -0.0196523913])
        beta3 = np.array([0.008973144, -0.10020531,  0.035812461,  0.0045485236])
        beta4 = np.array([-0.001156060,  0.01020321, -0.003818716, -0.0003689985])
        
        MvRR  = np.array([0.001502627, 0.0002023816, 4.707465e-05, 2.014207e-05])
        corrFact  = 1.000008
        meanMass  = 1.524476
        diffMass  = 78595.10163	

    elif numS == 2:
        beta0 = np.array([-0.033893697,  0.72349524,  0.032262337,  0.229835870])
        beta1 = np.array([0.565384632, -0.64468507,  0.429087497, -0.021658747])
        beta2 = np.array([-0.026136440,  0.53059249, -0.181136449,  0.097994130])
        beta3 = np.array([0.012468914, -0.13753241,  0.051527272, -0.027555700])
        beta4 = np.array([-0.001777315,  0.01309831, -0.005173147,  0.002782747])
        
        MvRR  = np.array([0.002005110, 0.0002673778, 7.106671e-05, 3.152397e-05])
        corrFact  = 1.000025
        meanMass  = 1.948059
        diffMass  = 29653.05953

    elif numS == 3:
        beta0 = np.array([-0.043228065,  0.97393090,  0.014513355,  0.360287349])
        beta1 = np.array([0.565105658, -0.87427089,  0.490702874, -0.163575517])
        beta2 = np.array([-0.025561801,  0.61317791, -0.204433409,  0.171747422])
        beta3 = np.array([0.011648239, -0.14907758,  0.053527995, -0.044669951])
        beta4 = np.array([-0.001532171,  0.01349895, -0.005008776,  0.004245513])
        
        MvRR  = np.array([0.002688456, 0.0003123977, 9.609067e-05, 4.079007e-05])
        corrFact  = 1.000089
        meanMass  = 2.376954
        diffMass  = 7870.61851

    elif numS == 4:
        beta0 = np.array([-0.091521425,  1.10056929, -0.001487005,  0.43455886])
        beta1 = np.array([0.659964385, -0.87136236,  0.538845132, -0.19351672])
        beta2 = np.array([-0.097607118,  0.54700360, -0.220722981,  0.16787935])
        beta3 = np.array([0.032790006, -0.12107593,  0.054711103, -0.03969725])
        beta4 = np.array([-0.003661898,  0.01012674, -0.004898160,  0.00348629])
        
        MvRR  = np.array([0.003429367, 0.0003440896, 1.218722e-04, 4.696665e-05])
        corrFact  = 1.000313
        meanMass  = 2.708084
        diffMass  = 1808.23104

    elif numS == 5:
        beta0 = np.array([-0.051732786,  1.37515374, -0.038979107,  0.562275988])
        beta1 = np.array([0.543672943, -1.15038799,  0.605477833, -0.324066003])
        beta2 = np.array([-0.010471838,  0.67223458, -0.243958456,  0.230450372])
        beta3 = np.array([0.005775558, -0.14776536,  0.057156580, -0.053389629])
        beta4 = np.array([-0.000745030,  0.01231632, -0.004884386,  0.004608342])
        
        MvRR  = np.array([0.003735255, 0.0003593458, 1.352894e-04, 5.205649e-05])
        corrFact  = 1.000989
        meanMass  = 2.934948
        diffMass  = 451.56855

    elif numS == 6:
        beta0 = np.array([-0.235720198,  1.271460962, -0.074691912,  0.532268779])
        beta1 = np.array([0.764850633, -0.795410368,  0.644406611, -0.197992545])
        beta2 = np.array([-0.109997374,  0.420266433, -0.243158716,  0.143211133])
        beta3 = np.array([0.024599540, -0.078510897,  0.052561006, -0.029599963])
        beta4 = np.array([-0.002115823,  0.005616518, -0.004201548,  0.002308974])
        
        MvRR  = np.array([0.004005769, 0.0003745374, 1.476534e-04, 5.550083e-05])
        corrFact  = 1.002899
        meanMass  = 3.101844
        diffMass  = 162.24158

    else:
        beta0 = np.array([0.275916431,  1.43225979,  0.315863733,  0.741687800])
        beta1 = np.array([0.084096306, -1.03919326,  -0.046926127, -0.553722694])
        beta2 = np.array([0.177867705,  0.62458122, 0.201073973,  0.394132137])
        beta3 = np.array([-0.023207940,  -0.14517056, -0.064888740, -0.101692499])
        beta4 = np.array([0.000249031,  0.01285476, 0.006830557,  0.009554752])
        
        MvRR  = np.array([0.005878334, 0.00153863, 0.0006399718, 0.0008042911])
        corrFact  = 1.006711
        meanMass  = 3.202
        diffMass  = 58.7158

    isoRatio = beta0 + beta1 * monoMass / 1000 + beta2 * (monoMass/1000)**2 + beta3 * (monoMass/1000)**3 + beta4 * (monoMass / 1000)**4

    if alpha == 0.05:
        k = 1.96
    elif alpha == 0.01:
        k = 3.29
    else:
        print("alpha = {} cannot be used, set alpha to 0.05".format(alpha))
        k = 1.96
        

    lwb = isoRatio - k * (MvRR)**0.5 * (corrFact + (monoMass/1000 - meanMass)**2 / diffMass)**0.5
    upb = isoRatio + k * (MvRR)**0.5 * (corrFact + (monoMass/1000 - meanMass)**2 / diffMass)**0.5


    output = pd.DataFrame({"fit":isoRatio, "lwb":lwb, "upb":upb})
    output = output.values
    return output