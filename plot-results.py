import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from helper import *
from juliacall import Main as jl

jl.seval('using Pkg; Pkg.add(url = "https://github.com/jbrea/MLPGradientFlow.jl.git")')
jl.seval('using MLPGradientFlow');
mg=jl.MLPGradientFlow

root_path="/Users/simsek/Desktop/TS"

def res_to_param(res):
    num_neurons = res["x"]["w1"].shape[0]
    input_dim = res["x"]["w1"].shape[1]
    angles = np.zeros((input_dim, num_neurons))
    norms = np.zeros(num_neurons)
    outgoing_weights = np.zeros(num_neurons)
    for i in range(num_neurons):
        incoming_weight = np.zeros(input_dim)
        for j in range(input_dim):
            incoming_weight[j] = res["x"]["w1"][i, j]
        angles[:, i] =  incoming_weight / np.linalg.norm(incoming_weight)
        norms[i] = np.linalg.norm(incoming_weight)
        outgoing_weights[i] = res["x"]["w2"][0, i]
    return angles, norms, outgoing_weights

num_student=30
regime="CC" # "CA" or "CC"
num_teacher=50
seed_id=1
file_name=root_path+"/data/erf-stud={:d}-teach={:d}-seed={:d}.pkl".format(num_student, num_teacher, seed_id)
res=mg.unpickle(file_name)

angles, norms, outgoing_weights = res_to_param(res)
neuron_ids = np.linspace(1, num_student, num_student)

fig, ax = plt.subplots(3, figsize=(3,6), height_ratios=[1, 1, 1], constrained_layout = True)
divnorm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

if(regime=="CC"):
    angles_perm, norms_perm, outgoing_weights_perm = permute_CC(angles, norms, outgoing_weights)
if(regime=="CA"):
    angles_perm, norms_perm, outgoing_weights_perm = permute_CA(angles, norms, outgoing_weights)
#plt.imshow(angles, cmap='RdBu', norm=divnorm)
#ax[0,1].rcParams["figure.figsize"] = (2.5,3)
H=num_teacher-num_student+1
im0=ax[0].imshow(angles_perm, cmap='RdBu', norm=divnorm)
ax[0].set_title('angles', fontsize=10)
fig.colorbar(im0, location='left')

ax[1].bar(neuron_ids, norms_perm, color ='brown', alpha=0.7,  width = 0.4, label='$r_i$')
ax[1].set_title('norms', fontsize=10)

ax[2].bar(neuron_ids, outgoing_weights_perm, color ='blue', alpha=0.7,  width = 0.4, label='$r_i$')
ax[2].set_title('outgoing weights', fontsize=10)

if(regime=="CC"):
    ax[1].axhline(y=1, linewidth=1, linestyle='dashed', color='gray')
    ax[2].axhline(y=1, linewidth=1, linestyle='dashed', color='gray')
    ax[2].axhline(y=-1, linewidth=1, linestyle='dashed', color='gray')
if(regime=="CA"):
    ax[1].axhline(y=1, linewidth=1, linestyle='dashed', color='gray')
    ax[1].axhline(y=1/np.sqrt(2*H-1), linewidth=1, linestyle='dashed', color='green')
    ax[2].axhline(y=1, linewidth=1, linestyle='dashed', color='gray')
    ax[2].axhline(y=-1, linewidth=1, linestyle='dashed', color='gray')
    ax[2].axhline(y=H, linewidth=1, linestyle='dashed', color='green')
    ax[2].axhline(y=-H, linewidth=1, linestyle='dashed', color='green')

plt.savefig(root_path+'/figs/box-stud={:d}-teach={:d}-seed={:d}.png'.format(num_student, num_teacher, seed_id), dpi=500)
