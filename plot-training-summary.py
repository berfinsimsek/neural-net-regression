import numpy as np
import os
from helper import *
import matplotlib.pyplot as plt
import juliacall as jc
from juliacall import Main as jl

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

jl.seval('using Pkg; Pkg.add(url = "https://github.com/jbrea/MLPGradientFlow.jl.git")')
jl.seval('using MLPGradientFlow');
mg=jl.MLPGradientFlow

root_path="/Users/simsek/Documents/GitHub/neural-net-regression"
if not os.path.exists(root_path+'/figs'):
    os.makedirs(root_path+'/figs')

num_teacher=50
num_seeds=20
gamma1=0.44
gamma21=0.52
gamma22=0.6
gamma3=0.8

gnorms=np.zeros((num_seeds, num_teacher))
losses=np.zeros((num_seeds, num_teacher))

optim_losses=np.zeros(num_teacher)


for num_student in range(1, num_teacher+1):
    print("num student:", num_student, " num teacher:", num_teacher)
    optim_losses[num_student-1]=optim_loss_erf(num_teacher-num_student+1)
    for seed_id in range(1,num_seeds+1):
         file_name=root_path+"/data/erf-stud={:d}-teach={:d}-seed={:d}.pkl".format(num_student, num_teacher, seed_id)
         res=mg.unpickle(file_name)
         gnorms[seed_id-1, num_student-1]=res['gnorm']
         losses[seed_id-1, num_student-1]=res['loss_curve'][-1]

plt.figure()
plt.plot(optim_losses, linestyle='--', linewidth=1, color='black', label='optimal CA')
plt.axvline(x=gamma1*num_teacher, linestyle='--', linewidth=1, color='lightcoral', label='$\gamma_1=0.44$')
plt.axvline(x=gamma21*num_teacher, linestyle='--', linewidth=1, color='red', label='$\gamma_2=0.5$')
plt.axvline(x=gamma22*num_teacher, linestyle='--', linewidth=1, color='red', label='$\gamma_2=0.6$')
plt.axvline(x=gamma3*num_teacher, linestyle='--', linewidth=1, color='darkred', label='$\gamma_3=0.8$')
for seed_id in range(num_seeds):
    plt.plot(losses[seed_id, :], 'o', markersize=3, alpha=0.2)
plt.legend(loc='upper right')
plt.xlabel('stud width')
plt.ylabel('losses')
plt.savefig(root_path+'/figs/losses_all.png', dpi=500)
plt.close()

plt.figure()
for seed_id in range(num_seeds):
    plt.plot(gnorms[seed_id, :], 'o', markersize=3, alpha=0.2)
plt.xlabel('stud width')
plt.ylabel('gnorms')
plt.savefig(root_path+'/figs/gnorms_all.png', dpi=500)
