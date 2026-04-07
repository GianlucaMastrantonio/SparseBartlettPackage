---
name: run-mcmc-sim
description: "Use when: scaffolding a new simulation script, creating a Julia test script, or setting up R post-processing for a new simulation run in SparseBartlettPersonal."
---

# run-mcmc-sim Skill

Scaffold a new end-to-end simulation: a Julia MCMC script and a matching R post-processing script that follow the project's established conventions.

## Workflow

### 1. Gather parameters

Ask the user (or infer from context) the simulation parameters:
- `pattern`: `Banded1`, `Banded2`, or `FullRandom`
- `k`: number of variables (e.g. `10`)
- `n`: sample size (e.g. `100`)
- `percmiss`: missingness fraction (e.g. `0.0`)
- `perczero`: sparsity level (e.g. `0.5`)
- `pi`: prior inclusion probability (e.g. `0.5`)
- `seeds`: vector of integer seeds (e.g. `[1,2,3,4,5]`)

### 2. Create the Julia simulation script

Place in `TestNewCode/` (or the user-specified folder). The script must:
- Load `SparseBartlettPackage` via `using SparseBartlettPackage`
- Use `NamedTuple` for MCMC iteration parameters: `(iter=…, burnin=…, thin=…)`
- Loop over seeds and save output to `.Rdata`-compatible files following the naming convention:
  ```
  NewEssModel_[Pattern]_LikeNormal_pi=[PI]_percmiss=[M]_n=[N]_k=[K]perczero=[Z]seed=[S]mcmc_out_lambda.Rdata
  ```
- Save output files to the `SparseBartlettPersonal/Rdata/` folder (absolute path from the Dropbox root).

### 3. Create the R post-processing script

Place in `SparseBartlettPersonal/funzioni confronto/` or `SparseBartlettPersonal/risultatisimulazioni/`. The script must:
- Load `.Rdata` files matching the naming convention above
- Generate traceplots and lambda plots (PDF) per seed
- Save plots to: `risultatisimulazioni/output/k=[K] percmiss=[M] pattern=[P] perczero=[Z]/`
- Follow the CRPS scoring pattern from `funzioni confronto/all plots.R`

### 4. Verify

- Confirm output directory paths exist (create with `dir.create(..., recursive=TRUE)` in R if needed)
- Remind the user to activate the Julia environment with `julia --project=.` from `SparseBartlettPackage/`
