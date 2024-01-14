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
epsilon1=2
epsilon2=0.1
gnorm_cut=1e-8

root_path="/Users/simsek/Desktop/TS"

for num_student in range(1, num_teacher+1):
    print("num student:", num_student, " num teacher:", num_teacher)

    def optim_loss_erf(x):
         return (x*np.arcsin(1/2) - x**2*np.arcsin(1/(2*x)))*2/np.pi

    plt.axhline(y=0, linestyle='dashed', color='gray')
    plt.axhline(y=optim_loss_erf(num_teacher-num_student+1), linestyle='dashed', color='red')

    gnorms=np.zeros(num_seeds)
    losses=np.zeros(num_seeds)

    for seed_id in range(1,num_seeds+1):
         file_name=root_path+"/data/erf-stud={:d}-teach={:d}-seed={:d}.pkl".format(num_student, num_teacher, seed_id)
         res=mg.unpickle(file_name)
         plt.plot(res['loss_curve'])
         gnorms[seed_id-1]=res['gnorm']
         losses[seed_id-1]=res['loss_curve'][-1]

    plt.ylabel('loss')
    plt.title('n={:02}'.format(num_student))
    plt.ylim(-epsilon2, optim_loss_erf(num_teacher-num_student+1)+epsilon1)
    plt.savefig(root_path+'/figs/loss_curves_{:d}stud_{:d}teach.png'.format(num_student, num_teacher), dpi=500)
    plt.close()

    plt.figure()
    plt.plot(gnorms, 'o')
    plt.xlabel('seed id')
    plt.ylabel('gnorm')
    plt.ylim(0, gnorm_cut)
    plt.savefig(root_path+'/figs/grnoms_{:d}stud_{:d}teach.png'.format(num_student, num_teacher), dpi=500)
    plt.close()
