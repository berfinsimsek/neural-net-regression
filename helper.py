import numpy as np

def optim_loss_erf(x):
         return (x*np.arcsin(1/2) - x**2*np.arcsin(1/(2*x)))*2/np.pi

def res_to_param(res, num_teacher):
    num_neurons = res["x"]["w1"].shape[0]
    #input_dim = res["x"]["w1"].shape[1]
    angles = np.zeros((num_teacher, num_neurons))
    norms = np.zeros(num_neurons)
    outgoing_weights = np.zeros(num_neurons)
    for i in range(num_neurons):
        incoming_weight = np.zeros(num_teacher)
        for j in range(num_teacher):
            incoming_weight[j] = res["x"]["w1"][i, j]
        angles[:, i] =  incoming_weight / np.linalg.norm(incoming_weight)
        norms[i] = np.linalg.norm(incoming_weight)
        outgoing_weights[i] = res["x"]["w2"][0, i]
    return angles, norms, outgoing_weights

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
