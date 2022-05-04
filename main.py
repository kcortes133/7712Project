import h5py
import numpy as np
import pandas as pd

import dataExploration, projection
import similarity

project = True
jaccardSim = False
manhattanDist = False

#file = 'CRC_GSE146771_10X_Cluster_AllDiffGenes_table (1).tsv'
#file = 'CRC_GSE136394_Cluster_AllDiffGenes_table.tsv'
#file = 'NSCLC_GSE99254_Cluster_AllDiffGenes_table.tsv'
#file = 'NSCLC_GSE139555_Cluster_AllDiffGenes_table.tsv'
#file = 'BRCA_GSE114727_10X_Cluster_AllDiffGenes_table.tsv'
file = 'BRCA_GSE110686_Cluster_AllDiffGenes_table.tsv'

cellTypes = dataExploration.readCellTypes()
scData = dataExploration.readSCdata(file)
genes = dataExploration.readGenes()
genes = dataExploration.readAllGenes()
exGenes = dataExploration.readExGenes()


cellTypeClusters = dataExploration.getGenesandES(cellTypes, genes)
scClusters = dataExploration.getGenesandES(scData, genes)
cellTypeNames = list(cellTypeClusters.index)
print(scClusters.index)
# 136394
#dataNames = ['CD8Tem', 'CD8Tcm', 'CD8Teff', 'CD8Tn']
# 146771
#dataNames = ['CD8Teff', 'CD8Tem', 'CD8Tex']
# NSCLC 99254 & 139555
#dataNames = ['CD8Tex', 'CD8Tcm', 'CD8Teff']
# BRCA 114727
#dataNames = ['CD8Tem', 'CD8Tcm', 'CD8Tex', 'CD8Tn']
# BRCA 110686
dataNames = ['CD8Tcm', 'CD8Tex', 'CD8Tem', 'CD8Teff']

scClusters = scClusters.loc[dataNames]

colsSC = scClusters.columns
rowsSC = scClusters.index
bt = scClusters.apply(lambda x: x>0.15)
nonZeroSC = bt.apply(lambda x: list(colsSC[x.values]), axis=1)

colsCT = cellTypeClusters.columns
rowsCT = cellTypeClusters.index
cellTypeClusters = cellTypeClusters.astype(float)
ct = cellTypeClusters.apply(lambda x: x>0.15)
nonZeroCT = ct.apply(lambda x: list(colsCT[x.values]), axis=1)

if manhattanDist:
    allData = cellTypeClusters.append(scClusters)
    allData = allData.replace(np.nan, 0)
    allData = allData.to_numpy(dtype='f')
    #allData = np.where(allData > 0.25, allData, 0)
    cellTypes = allData[0:len(cellTypeClusters), :]
    data = allData[len(cellTypeClusters):, :]


    for rowi in range(len(data)):
        print(rowsSC[rowi])
        mdists = {}
        for rowj in range(len(cellTypes)):
            mDist = similarity.manhattan(data[rowi], cellTypes[rowj])
            mdists[rowsCT[rowj]] = mDist
        print(dict(sorted(mdists.items(), key=lambda item: item[1], reverse=True)))





if jaccardSim:
    for i in range(len(nonZeroSC)):
        maxS = -1
        mostSim = 'None'
        ranking = {}
        for j in range(len(nonZeroCT)):
            omax = maxS
            dsim = similarity.jaccard(nonZeroSC[i],nonZeroCT[j])
            maxS = max(maxS, dsim)
            ranking[rowsCT[j]] = dsim
            if omax != maxS:
                mostSim = rowsCT[j]
        print('-------------')
        print(rowsSC[i])
        print(dataExploration.exMarkers(nonZeroSC[i], exGenes))
        print(mostSim, maxS)
        print(dict(sorted(ranking.items(), key=lambda item: item[1])))
        print(nonZeroSC[i])
        dataExploration.subtypeHist(ranking)


if project:
    allData = cellTypeClusters.append(scClusters)
    allData = allData.replace(np.nan, 0)

    colors = [0]*len(cellTypeClusters)
    colors.append(1)
    colors.append(2)
    colors.append(3)
    colors.append(4)

    allData = allData.to_numpy(dtype='f')
    allData = np.where(allData > 0.25, allData, 0)
    #similarity.nmf(allData, cellTypeNames+ dataNames)
    similarity.gowersDist(allData, cellTypeNames + dataNames)

    #projection.sphereProj(allData[0:len(cellTypeClusters), :], allData[len(cellTypeClusters):,:], cellTypeNames, dataNames)

