import numpy as np
import matplotlib.pyplot as plt
import juliacall as jc
from juliacall import Main as jl

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

jl.seval('using Pkg; Pkg.add(url = "https://github.com/jbrea/MLPGradientFlow.jl.git")')
jl.seval('using MLPGradientFlow');
mg=jl.MLPGradientFlow

num_teacher=50
num_seeds=20

max_losses=np.zeros(num_teacher)
min_losses=np.zeros(num_teacher)
max_gnorms=np.zeros(num_teacher)
min_gnorms=np.zeros(num_teacher)

root_path="/Users/simsek/Desktop/TS"

for num_student in range(1, num_teacher+1):
    print("num student:", num_student, " num teacher:", num_teacher)

    gnorms=np.zeros(num_seeds)
    losses=np.zeros(num_seeds)

    for seed_id in range(1,num_seeds+1):
         file_name=root_path+"/data/erf-stud={:d}-teach={:d}-seed={:d}.pkl".format(num_student, num_teacher, seed_id)
         res=mg.unpickle(file_name)
         gnorms[seed_id-1]=res['gnorm']
         losses[seed_id-1]=res['loss_curve'][-1]

    min_losses[num_student-1]=losses.min()
    max_losses[num_student-1]=losses.max()

    min_gnorms[num_student-1]=gnorms.min()
    max_gnorms[num_student-1]=gnorms.max()

plt.figure()
plt.plot(max_losses, 'o')
plt.plot(min_losses, 'o')
plt.xlabel('stud width')
plt.ylabel('losses')
plt.savefig(root_path+'/figs/losses_max_min.png', dpi=500)
plt.close()

plt.figure()
plt.plot(max_gnorms, 'o')
plt.plot(min_gnorms, 'o')
plt.xlabel('stud width')
plt.ylabel('losses')
plt.savefig(root_path+'/figs/gnorms_max_min.png', dpi=500)
