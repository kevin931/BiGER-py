import numpy as np
from PyCytoData import PyCytoData
from scipy.stats import rankdata

from numpy.typing import ArrayLike, NDArray


def preprocess_genelist(genelist: list[list[str]],
                        n_r: list[int],
                        n_u: list[int],
                        na_as_ties: bool=False):
    """Preprocess a genelist to generate a rank matrix for use in BiGER.

    This method preprocess a genelist and subsequently converts the ordered genelists
    into rankings for use in BiGER. This is the sister function to `preprocess_genelist`
    in R, but here, the interface is streamlined and improved.
    
    In general, we assume there are J studies with G genes in total. Each study can contain
    top-ranked, top-unranked, bottom ties, and missing genes. All top-ranked and top-unranked
    genes need to be included in `genelist`. If a cohort has no missing genes but has bottom ties,
    it is optional to include bottom ties in the `genelist`: in the case that they are not included,
    simply set `na_as_ties` to `True`. For a cohort with both bottom ties and missing genes,
    all bottom ties need to be included. 

    :param genelist: A meta-analysis of gene lists as a nested list. Each inner list consists of
    a study. For each inner list, the genes are ordered in a descending order (i.e. the first item is the most important).
    Each list needs to contain only a subset of all observed genes. Missing genes should not be included.
    :type genelist: list[list[str]]
    :param n_r: A list of length J as the number of top-ranked items in each gene list. If a study has no top-ranked genes,
    include 0. Included genes that are neither top-ranked or top-unranked are treated as bottom ties.
    :type n_r: list[int]
    :param n_u: A list of length J as the number of top-unranked items in each gene list. If a study has no top-unranked genes,
    include 0. Included genes that are neither top-ranked or top-unranked are treated as bottom ties.
    :type n_u: list[int]
    :param na_as_ties: Whether to treat missing genes from each list as bottom ties or simply as missing, defaults to False
    :type na_as_ties: bool, optional
    :return: A list of genes and an N by J integer array of genes' rankings in their corresponding study. Rankings start
    at 1. For all missing genes, a rank of -1 is used. 
    :rtype: tuple[list[str]]
    """
    
    # Find All genes 
    genes: set[str]|list[str] = set()
    for i in range(len(genelist)):
        genes = genes.union(set(genelist[i]))    
    genes = list(genes)
    
    # Rank matrix
    r = np.empty((len(genes), len(genelist)), dtype=int)
    gid: int
    for i in range(len(genelist)):
        if na_as_ties:
            r.fill(len(genelist[i])+1)
        else:    
            r.fill(-1)
        for j in range(len(genelist[i])):
            gid = genes.index(genelist[i][j])
            if j < n_r[i]:
                # Top Ranked
                r[gid, i] = j+1
            elif j < n_u[i]:
                # Top Unranked
                r[gid, i] = n_r[i] + 1
            else:
                # Bottom Ties
                r[gid, i] = n_r[i] + 1
    
    return genes, r

def preprocess_pycytodata(data: PyCytoData, score: ArrayLike, reverse: bool = False):
    """Interface to preprocess data from PyCytoData objects.

    _extended_summary_

    :param data: A PyCytoData object with appropriate channel names. We use `lineage_channels` by
    default, but if they are not present, all `channels` will be used.
    :type data: PyCytoData
    :param score: A numeric vector with the same length and order of the channels of interest from `data`.
    This can either be from existing ranking, p-value, or some other p-values. This is used to produce
    an order list of genes as output.
    :type score: ArrayLike
    :param reverse: Whether to reverse rank the `score`, defaults to False. If higher scores are better,
    then use the default. If lower scores are better (e.g. p-values or existing rankings), set `reverse`
    to `True`.
    :type reverse: bool, optional
    :raises ValueError: _description_
    :return: An ordered gene list from the PyCytoData object. 
    :rtype: list[str]
    """
    
    genes: list[str]
    score = np.array(score).flatten()
    
    if data.lineage_channels is None:
        genes = data.channels.tolist()
    else:
        genes = data.lineage_channels.tolist()
        
    if len(score) != len(genes):
        raise ValueError("The length of `score` must agree with the number of lineage channels in `data`.")    
    
    if reverse:
        genes = genes[rankdata(score, method="ordinal")] #type:ignore
    else:
        genes = genes[rankdata(-score, method="ordinal")] #type:ignore
    return genes
        
        
    