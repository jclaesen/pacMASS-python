import numpy as np
import calculateIsoRatio as iso 
import calculateNomMass as nom


def calculateAC(totalWeight, RR, ac, numS, ppm = 10, alpha = 0.05):
    
    weight = np.array([12, 1.0078250321, 14.0030740052, 15.9949146, 31.97207070])   # weight [C, H, N, O, S]
    tolerance = ppm * totalWeight / 10**6

    AC = ac[ac[:,4]==numS]
    RR2= RR[ac[:,4]==numS]
   
    # calculate prediction interval for isoRatios
    estimateRR = iso.calculateIsoRatio(numS, totalWeight, alpha)     # columns: fit, lwb, upb 
    
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
    nominalMass = nom.calculateNomMass(numS, totalWeight, alpha)
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

