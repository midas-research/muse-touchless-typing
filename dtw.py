#code inspired from https://github.com/bguillouet/traj-dist/blob/master/traj_dist/pydist/dtw.py

import numpy as np
from scipy import spatial
from numpy import dot,inf
from numpy.linalg import norm
import hp
import editdistance

def eucl_dist(x, y):
    """
    L2-norm between point x and y
    Parameters.
    
    Arguments:
        x: numpy_array
        y: numpy_array
    Returns
        dist(float): L2-norm between x and y
    """
    
    dist = np.linalg.norm(x - y)
    return dist

def cosine_sim(v1,v2,eps=1e-5):
    """
    1-cosine_similarity between vectors v1 and v2
    Parameters.
    
    Arguments:
        v1: list
        v2: list
    Returns
        cosine_sim(float): 1-cosine_similarity between v1 and v2.
    """
    
    v1 = np.array(v1)
    v2 = np.array(v2)
    cosine_sim = 1-dot(v1, v2)/(norm(v1)*norm(v2)+eps)
    return cosine_sim

def cmp(s1,s2):
    return s1[0]==s2[0] and s1[1]==s2[1]

def modified_dtw(t0, t1):
    """
    The Modified Dynamic-Time Warping distance between trajectory t0 and t1.
    https://www.hindawi.com/journals/mpe/2018/2404089/
    
    Arguments:
        t0: len(t0)x2 numpy_array.
        t1: len(t1)x2 numpy_array.
    Returns:
        dtw(float): The Dynamic-Time Warping distance between trajectory t0 and t1.
    """
    
    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = float('inf')
    C[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            C[i, j] =  min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j])
            ed = eucl_dist(t0[i - 1], t1[j - 1])
            cs = 0
            alpha = 0
            if not(i==1 or j==1 or i==n0 or j==n1):  #for considering both near and neighbour elements
                v1 = ((t0[i-1]-t0[i-2])+(t0[i]-t0[i-2])/2)/2
                v2 = ((t1[j-1]-t1[j-2])+(t1[j]-t1[j-2])/2)/2
                cs = cosine_sim(v1,v2)
                alpha = 0.5

            C[i,j]+=alpha*cs*ed+(1-alpha)*ed

    dtw = C[n0, n1]
    return dtw

#mapping clusters to 2d coordinates with 5 as (0,0)
clust_to_cord = {
'1' : [-1.0,1.0],
'2' : [0,1.0],
'3' : [1.0,1.0],
'4' : [-1.0,0],
'5' : [0,0],
'6' : [1.0,0],
'7' : [-1.0,-1.0],
'8' : [0,-1.0],
'9' : [1.0,-1.0]
}

def get_dtw(s1,s2):
    """
    Return dtw for a pair sequences.
    
    Arguments:
        s1(string): first sequence.
        s2(string): second sequence.
    Returns:
        float: modified dtw between s1 and s2.
    """
    
    l1 = [];l2 = []
    for i in s1:
        l1.append(clust_to_cord[i])
    for i in s2:
        l2.append(clust_to_cord[i])
    l1 = np.array(l1);l2=np.array(l2)
    return modified_dtw(l1,l2)

def dtw_batch(seqs,target_seqs):
    """
    Return dtw for a batch.
    
    Arguments:
        seqs: len(t0)x2 numpy_array.
        target_seqs: len(t1)x2 numpy_array.
    Returns:
        avg_dtw(float): dtw for batch.
    """
    
    assert len(seqs)==len(target_seqs),"batching error in dtw_batch"
    avg_dtw = 0
    corrected_sequences = []
    for seq,tseq in zip(seqs,target_seqs):
        avg_dtw+=get_dtw(seq,tseq)
    avg_dtw = avg_dtw/len(seqs)
    return avg_dtw

if __name__=='__main__':
    s1 = "346222"
    s2 = "3462"
    print (get_dtw(s1,s2))
    #     Original:  364
    # Decoded:  63641
    # Best match:  6371


