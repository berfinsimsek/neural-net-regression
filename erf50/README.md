Run the script `erf_sims.jl` in this folder with

```bash
julia -pNUM_PROCESSES --project=. erf_sims.jl SEEDS N_TEACHER_NEURONS
# for example
# julia -p48 --project=. erf_sims.jl 1:40 50
```
If gradient flow did not converge for a given configurations within the given time limit,
one can restart gradient flow from the result `res["x"]`.
