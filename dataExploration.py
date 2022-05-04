import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt


def readCellTypes():
    df = pd.read_csv('science.abe6474_table_s3.csv')
    df = df.rename(columns=df.iloc[0])
    df.drop(df.index[0], inplace=True)
    print(df.columns)
    # geneSymbol comb.ES
    df = df[['cluster.name', 'geneSymbol', 'comb.ES']]
    df = df.rename(columns={'geneSymbol': 'Gene', 'comb.ES': 'ES', 'cluster.name': 'Cluster'})
    return df


def readSCdata(file):
    df = pd.read_csv(file, sep='\t')
    # Gene log2FC
    df = df[['Celltype (minor-lineage)', 'Gene', 'log2FC']]
    df = df.rename(columns={'log2FC': 'ES', 'Celltype (minor-lineage)': 'Cluster'})
    df = df.astype({'ES':'float'})
    return df


def readGenes():
    df = pd.read_csv('genes.txt')
    return df['genes'].values.tolist()


def readExGenes():
    df = pd.read_csv('exGenes.txt')
    print(df.columns)
    return {k: g['Gene'].tolist() for k,g in df.groupby("Subtype")}

def readAllGenes():
    df = pd.read_csv('science.abe6474_table_s3.csv')
    df = df.rename(columns=df.iloc[0])
    df.drop(df.index[0], inplace=True)
    return list(set(df['geneSymbol']))

def getGenesandES(df, genes):
    df = df[df['Gene'].isin(genes)]
    geneES = {}
    for index, row in df.iterrows():
        cluster = row['Cluster']
        gene = row['Gene']
        es = row['ES']

        if cluster in geneES:
            geneES[cluster][gene] = es
        else:
            geneES[cluster] = {gene: es}

    dfG = pd.DataFrame.from_dict(geneES, orient='index')
    dfG = dfG.replace(np.nan, 0)
    return dfG



def exMarkers(genes, exGenes):
    exSim = {}

    for ex in exGenes:
        jsim = jaccard(exGenes[ex], genes)
        intersectionGenes = list(set(exGenes[ex]).intersection(genes))
        intersection = len(intersectionGenes)
        exSim[ex] = (jsim, intersection)
        print(ex, intersectionGenes)
        print(intersectionGenes)

    return exSim


def subtypeHist(subtypeSims):
    width = 0.9
    pos = np.arange(len(subtypeSims.keys()))
    ax = plt.axes()
    ax.set_xticks(pos + (width/2))
    ax.set_xticklabels(subtypeSims.keys())
    ax.invert_yaxis()
    plt.bar(subtypeSims.keys(), subtypeSims.values(), color='g')
    #plt.xticks(rotation='90')
    plt.show()

