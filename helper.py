import numpy as np

def permute_CA(angles, norms, outgoing_weights):
    n = angles.shape[1]
    d = angles.shape[0]
    i = 0
    while(i < n):
        angles_swap = angles.copy()
        norms_swap = norms.copy()
        outgoing_weights_swap = outgoing_weights.copy()
        if(np.abs(angles[:, i]).max() < 0.95):
            angles_swap[:, i] = angles[:, n-1]
            angles_swap[:, n-1] = angles[:, i]
            norms_swap[i] = norms[n-1]
            norms_swap[n-1] = norms[i]
            outgoing_weights_swap[i] = outgoing_weights[n-1]
            outgoing_weights_swap[n-1] = outgoing_weights[i]
            angles = angles_swap.copy()
            norms = norms_swap.copy()
            outgoing_weights = outgoing_weights_swap.copy()
        i+=1
    for j in range(n-1):
        angles_swap_swap = angles_swap.copy()
        argmax_j = np.abs(angles_swap[:, j]).argmax()
        #print(j, argmax_j)
        angles_swap_swap[argmax_j, :] = angles_swap[j, :]
        angles_swap_swap[j, :] = angles_swap[argmax_j, :]
        angles_swap = angles_swap_swap.copy()
    return angles_swap_swap, norms_swap, outgoing_weights_swap #angles_swap_swap

def permute_CC(angles, norms, outgoing_weights):
    n = angles.shape[1]
    d = angles.shape[0]
    for j in range(n):
        angles_swap = angles.copy()
        argmax_j = np.abs(angles_swap[:, j]).argmax()
        #print(j, argmax_j)
        angles_swap[argmax_j, :] = angles[j, :]
        angles_swap[j, :] = angles[argmax_j, :]
        angles = angles_swap.copy()
    return angles_swap, norms, outgoing_weights
