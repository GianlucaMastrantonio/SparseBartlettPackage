# SparseBartlettPackage — Copilot Instructions

## Project Overview

Julia package for Bayesian estimation of **sparse precision matrices** via the Bartlett decomposition. Uses MCMC (HMC, Metropolis, reversible-jump) to sample posterior distributions over sparse Cholesky factors.

This is a **multi-root workspace** with three components:
- `SparseBartlettPackage/` — the Julia package (this repo, version-controlled)
- `SparseBartlettPersonal/` — R post-processing and comparison scripts
- `SparsePrecisionPatterns - GM/` — LaTeX paper source

## Build and Test

```julia
# Activate and instantiate
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Load interactively
julia --project=.
using SparseBartlettPackage
```

> Tests in `test/runtests.jl` are currently minimal stubs — add tests alongside any new functionality.

## Architecture

| File | Role |
|------|------|
| `src/types.jl` | Core structs (e.g. `order_lambda_V3`). All structs use keyword-only inner constructors. |
| `src/mcmc.jl` | Main MCMC loop for continuous data |
| `src/mcmc_glm.jl` | MCMC for generalized linear model variant |
| `src/hmc.jl` | HMC infrastructure (dual averaging, NUTS: `HMC_DualAv`, `HMC_DualAv_NoUTurn`) |
| `src/sample_q.jl` | HMC sampler for the Cholesky factor `q` |
| `src/sample_q_metropolis.jl` | Metropolis sampler for `q` |
| `src/sample_glm_q.jl` | GLM variant of `q` sampler |
| `src/sample_z.jl` | Sampler for latent variables `z` |
| `src/sample_mu.jl` | Sampler for mean vector `mu` |
| `src/det_prop.jl` | Determinant proposals |
| `src/change_order.jl` | Column/row permutation logic |
| `src/functions.jl` | Shared utility functions |

**Exported API:** `mcmc`, `mcmc_glm`, `SparseRJ`, `GeneralS`, `HMCRJ`, `BaseHMC`, `GenSparse`, `HMC_DualAv_NoUTurn`, `HMC_DualAv`, `compute_unique_permutations`

## Conventions

- **Structs use keyword constructors** (`function MyStruct(; kwargs...)`). Always match this pattern when adding new types.
- **`Float64` / `Int64` are explicit** in all struct fields and function signatures — avoid `Any` or implicit types.
- **All MCMC functions use keyword arguments** with `NamedTuple` for iteration parameters: `(iter=…, burnin=…, thin=…)`.
- **Autodiff backends:** `ForwardDiff` and `Enzyme` are both present — check which is active per sampler before adding gradient code.
- **`ToggleableAsserts`** is used for debug-mode assertions that can be disabled in production. Use `@toggled_assert` instead of plain `@assert` for performance-critical paths.
- `Zygote` is listed as a dependency but currently unused (commented-out code). Do not add new `Zygote`-based gradients without confirming it is being re-introduced.

## R Analysis (`SparseBartlettPersonal/`)

- Results are `.Rdata` files following the naming pattern:  
  `NewEssModel_[Pattern]_LikeNormal_pi=[PI]_percmiss=[M]_n=[N]_k=[K]perczero=[Z]seed=[S]mcmc_out_lambda.Rdata`
- Patterns include: `Banded1`, `Banded2`, `FullRandom`
- Plots are saved to `risultatisimulazioni/output/k=[K] percmiss=[M] pattern=[P]/`
- `funzioni confronto/all plots.R` is the main script for loading results and generating plots

## Known Issues / Pitfalls

- `docs/make.jl` references the old package name `SparseIW` — it does not build correctly. Update before generating docs.
- `README.md` badge links also reference the old repo name `SparseIW.jl`.
- Some files are commented out in the module (`sample_glm_z.jl`, `hmc_functions.jl`) — they are incomplete stubs, do not include them.
