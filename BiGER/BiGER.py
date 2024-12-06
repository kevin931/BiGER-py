import numpy as np
from scipy.stats import truncnorm, gamma, norm
from copy import deepcopy

from numpy.typing import NDArray
from typing import List

def _get_ranked_index(r: NDArray[np.int64],
                      n_ranked: NDArray[np.int64]):
    ranked_index: List[NDArray[np.int64]] = []
    i: int
    for i in np.arange(r.shape[1]):
        if n_ranked[i] != 0:
            indices: NDArray[np.int64] = np.zeros(n_ranked[i], dtype=np.int64)
            for j in np.arange(n_ranked[i]):
                indices[j] = (r[:, i] == j + 1).nonzero()[0][0]
            ranked_index.append(indices)
        else:
            ranked_index.append(np.array([], dtype=np.int64))
    return ranked_index


def _get_unranked_index(r: NDArray[np.int64],
                        n_ranked: NDArray[np.int64],
                        n_unranked: NDArray[np.int64]):
    unranked_index: List[NDArray[np.int64]] = []
    for i in np.arange(r.shape[1]):
        if n_unranked[i] != 0:
            indices: NDArray[np.bool_] = r[:,i] == n_ranked[i] + 1
            unranked_index.append(indices.nonzero()[0])
        else:
            unranked_index.append(np.array([], dtype=np.int64))
    return unranked_index

def _get_bottom_index(r: NDArray[np.int64],
                      n_ranked: NDArray[np.int64],
                      n_unranked: NDArray[np.int64]):
    bottom_index: List[NDArray[np.int64]] = []
    for i in np.arange(r.shape[1]):
        n_non_na: int = np.sum(np.invert(np.isnan(r[:,i])))
        if n_ranked[i] + n_unranked[i] < n_non_na:
            indices: NDArray[np.bool_] = r[:,i] == (n_ranked[i] + n_unranked[i] + 1)
            bottom_index.append(indices.nonzero()[0])
        else:
            bottom_index.append(np.array([], dtype=np.int64))
    return bottom_index


def _get_non_na(r):
    non_na: NDArray[np.int64] = np.zeros(r.shape, dtype="int64")
    for i in np.arange(r.shape[0]):
        non_na[i] = np.isfinite(r[i])
    return non_na


def _update_mu(n_items, non_na_row, m_mu1, s2_mu, e_sigma2_inv, e_w):
        
    for g in np.arange(n_items):
        # ranked_study: NDArray[np.int64] = np.isfinite(r[g,:])
        denominator: NDArray[np.float64] = np.sum(e_sigma2_inv[non_na_row[g]])+1.0
        m_mu1[g] = np.sum(e_sigma2_inv[non_na_row[g]]*e_w[g, non_na_row[g]])/denominator
        s2_mu[g] = 1/denominator
        
    return m_mu1, s2_mu


def _update_s2(n_lists, a, b, non_na_col, alpha, beta, e_w, e_w2, m_mu, s2_mu, e_sigma2_inv):
    for j in np.arange(n_lists):
        # ranked_item: NDArray[np.int64] = np.isfinite(r[:,j])
        a[j] = 0.5*np.nansum(non_na_col[j]) + alpha
        b[j] =  np.sum(0.5*e_w2[non_na_col[j],j] - e_w[non_na_col[j],j]*m_mu[non_na_col[j]] + 0.5*(s2_mu[non_na_col[j]] + m_mu[non_na_col[j]]**2)) + beta
        e_sigma2_inv[j] = a[j]/b[j]
        
    return a, b, e_sigma2_inv


def _update_w(n_lists, n_ranked, n_unranked, non_na_col, ranked_index, unranked_index, bottom_index, e_w, e_w2, m_mu, e_sigma2_inv):
    for s in np.arange(n_lists):
        upper: float
        lower: float
        n_non_na: int = np.sum(non_na_col[s])
        index = np.arange(n_ranked[s], dtype=np.int64) + 1 
        np.random.shuffle(index)
        
        # Top Ranked Items            
        if (n_ranked[s] > 0):
            for g in np.arange(n_ranked[s], dtype=np.int64):
                # The first item
                i = index[g]
                if index[g] == 1:
                    lower = e_w[ranked_index[s][1], s] #type: ignore
                    upper = np.inf
                # The last item
                elif index[g] == n_ranked[s]:
                    upper = e_w[ranked_index[s][index[g]-2], s] #type: ignore
                    if n_ranked[s] < n_non_na:
                        if (n_unranked[s] > 0):
                            lower = np.max(e_w[unranked_index[s],s]) #type: ignore
                        else:
                            lower = np.max(e_w[bottom_index[s],s]) #type: ignore
                    else:
                        lower = -np.inf
                # In-Between
                else:
                    lower = e_w[ranked_index[s][index[g]],s] #type: ignore
                    upper = e_w[ranked_index[s][index[g]-2],s] #type: ignore
                    
                    
                e_w[ranked_index[s][index[g]-1],s] = truncnorm.moment(1,
                                                                        a = (lower-m_mu[ranked_index[s][index[g]-1]])/np.sqrt(1/e_sigma2_inv[s]),
                                                                        b = (upper-m_mu[ranked_index[s][index[g]-1]])/np.sqrt(1/e_sigma2_inv[s]),
                                                                        loc = m_mu[ranked_index[s][index[g]-1]],
                                                                        scale = np.sqrt(1/e_sigma2_inv[s]))
                e_w2[ranked_index[s][index[g]-1],s] = truncnorm.moment(2,
                                                                        a = (lower-m_mu[ranked_index[s][index[g]-1]])/np.sqrt(1/e_sigma2_inv[s]),
                                                                        b = (upper-m_mu[ranked_index[s][index[g]-1]])/np.sqrt(1/e_sigma2_inv[s]),
                                                                        loc = m_mu[ranked_index[s][index[g]-1]],
                                                                        scale = np.sqrt(1/e_sigma2_inv[s]))
                

        # Unranked Items
        if n_unranked[s] > 0:
            # breakpoint()
            if n_ranked[s] + n_unranked[s] < n_non_na:
                lower = np.max(e_w[bottom_index[s], s]) #type: ignore
            else:
                lower = -np.inf
            
            if n_ranked[s] == 0:
                upper = np.inf
            else:
                upper = e_w[ranked_index[s][n_ranked[s]-1],s] #type: ignore
                
            for j in unranked_index[s]:
                e_w[j,s] = truncnorm.moment(1,
                                            a = (lower-m_mu[j])/np.sqrt(1/e_sigma2_inv[s]),
                                            b = (upper-m_mu[j])/np.sqrt(1/e_sigma2_inv[s]),
                                            loc = m_mu[j],
                                                scale = np.sqrt(1/e_sigma2_inv[s]))
                e_w2[j,s] = truncnorm.moment(2,
                                                a = (lower-m_mu[j])/np.sqrt(1/e_sigma2_inv[s]),
                                                b = (upper-m_mu[j])/np.sqrt(1/e_sigma2_inv[s]),
                                                loc = m_mu[j], scale = np.sqrt(1/e_sigma2_inv[s]))
                
        
        # Bottom Ties
        if n_ranked[s] + n_unranked[s] < n_non_na:
            # breakpoint()
            lower = -np.inf
            if (n_unranked[s] > 0):
                upper = np.min(e_w[unranked_index[s], s]) #type: ignore
            else:
                upper = e_w[ranked_index[s][n_ranked[s]-1],s] #type: ignore
                
            if lower > upper:
                print("Error")
                
            for j in bottom_index[s]:
                e_w[j,s] = truncnorm.moment(1,
                                            a = (lower-m_mu[j])/np.sqrt(1/e_sigma2_inv[s]),
                                            b = (upper-m_mu[j])/np.sqrt(1/e_sigma2_inv[s]),
                                            loc = m_mu[j],
                                            scale = np.sqrt(1/e_sigma2_inv[s]))
                e_w2[j,s] = truncnorm.moment(2,
                                                a = (lower-m_mu[j])/np.sqrt(1/e_sigma2_inv[s]),
                                                b = (upper-m_mu[j])/np.sqrt(1/e_sigma2_inv[s]),
                                                loc = m_mu[j],
                                                scale = np.sqrt(1/e_sigma2_inv[s]))
                    
    return e_w, e_w2


def BiGER_VI(r: NDArray[np.int64],
             n_ranked: NDArray[np.int64],
             n_unranked: NDArray[np.int64],
             max_iter: int,
             alpha: float,
             beta: float,
             e_w0: NDArray[np.float64],
             e_w20: NDArray[np.float64],
             e_sigma2_inv0: NDArray[np.float64]):
    
    # Initializations
    n_items: int = r.shape[0]
    n_lists: int = r.shape[1]
    convergence: NDArray[np.float64] = np.empty(max_iter - 1, dtype=np.float64)
    m_mu: NDArray[np.float64] = np.empty(n_items, dtype=np.float64)
    m_mu1: NDArray[np.float64] = np.empty(n_items, dtype=np.float64)
    s2_mu: NDArray[np.float64] = np.empty(n_items, dtype=np.float64)
    a: NDArray[np.float64] = np.empty(n_items, dtype=np.float64)
    b: NDArray[np.float64] = np.empty(n_items, dtype=np.float64)
    e_w: NDArray[np.float64] = np.empty((n_items,n_lists), dtype=np.float64)
    e_w2: NDArray[np.float64] = np.empty((n_items,n_lists), dtype=np.float64)
    e_sigma2_inv: NDArray[np.float64] = np.empty(n_lists, dtype=np.float64)
    
    e_w = e_w0
    e_w2 = e_w20
    e_sigma2_inv = e_sigma2_inv0
    
    # Get Indices
    ranked_index = _get_ranked_index(r, n_ranked)
    unranked_index = _get_unranked_index(r, n_ranked, n_unranked)
    bottom_index = _get_bottom_index(r, n_ranked, n_unranked)
    
    # Non-NA Index
    non_na_row: NDArray[np.int64] = _get_non_na(r)
    non_na_col: NDArray[np.int64] = np.transpose(non_na_row)

    for iter in np.arange(max_iter):
        if iter % 10 == 0:
            print(iter)
        
        # Update mu        
        m_mu1, s2_mu = _update_mu(n_items, non_na_row, m_mu1, s2_mu, e_sigma2_inv, e_w)
        if iter > 0:
                convergence[iter-1] = np.mean(m_mu1-m_mu)
        m_mu = deepcopy(m_mu1)
        
        # Update sigma
        a, b, e_sigma2_inv = _update_s2(n_lists, a, b, non_na_col, alpha, beta, e_w, e_w2, m_mu, s2_mu, e_sigma2_inv)
        
        # Update W
        e_w, e_w2 = _update_w(n_lists, n_ranked, n_unranked, non_na_col, ranked_index, unranked_index, bottom_index, e_w, e_w2, m_mu, e_sigma2_inv)
        
    return m_mu, convergence