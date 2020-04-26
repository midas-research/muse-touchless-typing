import torch 
import torch.nn as nn
import hp

def check_loss(loss, loss_value):
    """
    Check that warpctc loss is valid and will not break training
    
    Returns:
        (boolean,string): Return if loss is valid, and the error in case it is not
    """
    
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error


def convert_to_strings(target_sequence):
    """
    Given a list of target sequences, maps the indexes to the 
    corresponding character.
    
    Arguments:
        target_sequence: A list of target sequences containing indexes
                            of the cluster.       
    Returns:
        list: A list of target sequences,where each index to mapped to 
                a cluster.
    """
    
    cluster_sequence = []
    
    for seq in target_sequence:
        cluster_sequence.append(''.join([str(i) for i in seq]))
        
    return cluster_sequence


def check_sequences(seqs):
    """
    Check if each sequences is present in the defined
    lexicon.
    
    Arguments:
        seqs(list): A list of decoded sequences.       
    Returns:
        boolean: whether all sequences are in lexicon or 
                    not.
    """
    
    for seq in seqs:
        if seq not in hp.lexicon:
            return False
        
    return True
