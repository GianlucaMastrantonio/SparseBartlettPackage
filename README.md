# SparseBartlettPackage.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)

Bayesian estimation of **sparse precision matrices** (Gaussian graphical models) with the
**Sparse-Bartlett (SB)** distribution. SB is a hierarchical prior defined on precision matrices
with an *arbitrary* zero pattern.

> ⚠️ **Work in progress.** This is an early, unfinished release that accompanies the paper below.
> The interface will change in future versions. See [Current limitations](#current-limitations).

---

## The model in a nutshell

The observations are $\mathbf y_i \sim N_p(\boldsymbol\mu_i, \Lambda^{-1})$, $i=1,\dots,n$.
A graph $\mathcal G$ is encoded by a binary symmetric matrix $Z$, where
$z_{jk}=0 \iff \lambda_{jk}=0$.

- **Ordered Sparse-Bartlett (OSB).** Write $\Lambda = QQ^\top$ (Cholesky, $Q$ lower-triangular).
  The free entries of $Q$ are drawn as in the Bartlett decomposition of a Wishart$(\nu, S)$.
  The degrees of freedom of each diagonal element are reduced to the number of free entries in
  its column. The constrained entries of $Q$ are set deterministically so that the zeros of
  $\Lambda$ hold exactly. Free coordinates and $\mathcal P_p(\mathcal G)$ are in bijection, so no
  intractable normalising constant appears.
  A full $Z$ gives the Wishart. A diagonal $Z$ gives independent Gammas.
- **Sparse-Bartlett (SB).** The OSB depends on the ordering of the variables. SB mixes the OSB
  over a uniformly distributed permutation $D$, which makes the prior permutation-equivariant.
- **Graph prior.** Spike-and-slab on the edges, $z_{jk} \sim \text{Bernoulli}(\pi_{jk})$.
- **Mean.** $\boldsymbol\mu_i = X_i \boldsymbol\beta$, with a Gaussian prior on $\boldsymbol\beta$.

### MCMC

The sampler works on a fixed-dimensional augmented space $(\tilde B, Z, D)$. Each iteration:

1. updates the latent Bartlett factor $\tilde B$ element-wise with adaptive random-walk Metropolis
   (on the log scale for the diagonal);
2. updates each $z_{jk}$ by a Gibbs (Bernoulli) step;
3. updates the mean coefficients $\boldsymbol\beta$ (Gaussian full conditional);
4. updates the permutation $D$ jointly with $\tilde B$. The proposal is a swap of two variables
   with a deterministic map that leaves $\Lambda$ unchanged, so the likelihood cancels;
5. optionally imputes missing values from their Gaussian full conditional.

---

## Installation

The package is not registered yet. Install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/GianlucaMastrantonio/SparseBartlettPackage")
```

For development, use a local clone:

```julia
Pkg.develop(path = "/path/to/SparseBartlettPackage")
```

---

## Quick start

```julia
using SparseBartlettPackage
using Distributions, LinearAlgebra, Random

Random.seed!(1)

# ---- simulated data: banded (tridiagonal) precision ---------------------
k, n = 10, 100                      # k = number of variables (p in the paper)
Λtrue = Matrix(1.0I, k, k)
for j in 1:(k - 1)
    Λtrue[j + 1, j] = Λtrue[j, j + 1] = 0.4
end
Y = rand(MvNormal(zeros(k), Symmetric(inv(Λtrue))), n)   # k × n  (variables in rows!)

# ---- prior and MCMC settings -------------------------------------------
nm = k * (k - 1) ÷ 2                # number of off-diagonal elements
ν  = 3.0                            # SB degrees of freedom (paper notation)
ψ2 = 1.0                            # scale: S = ψ² I

out = mcmc(
    ydata            = Y,
    iterations       = (iter = 10_000, burnin = 8_000, thin = 2),
    covariates       = ones(Float64, k, 1, n),          # common intercept
    prior_lambda     = Wishart(ν + k - 1, ψ2 * Matrix(1.0I, k, k)),
    prior_mu         = Normal(0.0, sqrt(1000.0)),
    m_init           = zeros(nm),
    c_init           = ones(k),
    beta_init        = zeros(1),
    z_init           = ones(k, k),                       # start from the full graph
    prob_zeta        = fill(0.5, k, k),                  # π_jk = 0.5
    model_sparse     = GenSparse_V2(fill(1.0 / nm, nm), Normal(0.0, 1.0)),
    model_psi        = GeneralS(),
    type_mcmc_lambda = HMC_DualAv_NoUTurn(; epsilon = 0.1, L = 10, M = 10,
                                          restart = false, delta = 0.5,
                                          iter_adapt = 1_000),
    iter_start_zeta  = 5,
)
```

### Posterior summaries

```julia
# posterior mean of the precision matrix
Λhat = dropdims(mean(out.lambda_out, dims = 1), dims = 1)

# posterior edge-inclusion probabilities (read the LOWER triangle)
pip = dropdims(mean(out.zmat_out, dims = 1), dims = 1)

# median-probability graph
edges = [(j, l) for l in 1:(k - 1) for j in (l + 1):k if pip[j, l] > 0.5]

# number of edges along the chain
n_edges = [sum(out.zmat_out[s, j, l] for l in 1:(k - 1) for j in (l + 1):k)
           for s in axes(out.zmat_out, 1)]
```

### Missing data

Pass the positions of the missing entries as an `n_miss × 2` integer matrix of
`(variable, observation)` pairs, i.e. `(row, column)` indices of `ydata`. The corresponding
entries of `ydata` must hold a **finite starting value**, such as `0.0` or a random draw, not
`NaN` or `missing`.

```julia
miss = [3 10;
        7 42]                        # y[3,10] and y[7,42] are missing
for r in eachrow(miss)
    Y[r[1], r[2]] = 0.0              # starting value
end

out = mcmc(; ydata = Y, missing_index = miss, #= ...other arguments as above... =#)

out.missing_out                      # n_samples × n_miss posterior draws of the missing values
```

---

## The `mcmc` function

All arguments are keyword arguments. Types are strict: use `Float64` and `Int64` as shown.

| Argument | Type | Description |
|---|---|---|
| `ydata` | `Matrix{Float64}` (k × n) | Data, **variables in rows, observations in columns**. Standardising each variable is recommended. |
| `iterations` | `NamedTuple` `(iter, burnin, thin)` | Total iterations, burn-in, thinning. The number of saved draws is `round((iter - burnin) / thin)`. |
| `covariates` | `Array{Float64,3}` (k × q × n) | `covariates[:, :, i]` is the design matrix $X_i$ of observation *i*, so $\boldsymbol\mu_i = X_i\boldsymbol\beta$. Use `ones(k, 1, n)` for a single common intercept. For one intercept per variable, set each slice to `Matrix(1.0I, k, k)`. |
| `prior_lambda` | `Wishart` | Gives ν and the scale *S*: `Wishart(ν + k - 1, S)`. The package uses the paper's ν, so pass `ν + k - 1` (the Distributions.jl convention). **Currently *S* must be ψ²I.** |
| `prior_mu` | `Normal` | Prior on each element of β, applied independently. |
| `m_init` | `Vector{Float64}` (k(k-1)/2) | Initial off-diagonal entries of the latent Bartlett factor $\tilde B$, column-wise. |
| `c_init` | `Vector{Float64}` (k) | Initial diagonal entries of $\tilde B$ (must be > 0). |
| `beta_init` | `Vector{Float64}` (q) | Initial regression coefficients. |
| `z_init` | `Matrix{Float64}` (k × k) | Initial graph. Only the strict lower triangle is used: `1.0` = edge (free entry), `0.0` = zero in Λ. |
| `prob_zeta` | `Matrix{Float64}` (k × k) | Prior inclusion probabilities π_jk = P(z_jk = 1). Only the strict lower triangle is used. |
| `model_sparse` | `GenSparse_V2` | Sparsity model. `GenSparse_V2(prob_vec, distr)`: `distr` is the prior of the off-diagonal Bartlett variables $\tilde b_{jk}$. `Normal(0, 1)` gives the SB of the paper; `Laplace(0, 1/√2)` is a heavier-tailed alternative with unit variance. `prob_vec` (length k(k-1)/2) is currently not used. |
| `model_psi` | `GeneralS()` | Placeholder for the scale model (currently not used). |
| `type_mcmc_lambda` | `<: TypeMCMC` | Sampler settings object, e.g. `HMC_DualAv_NoUTurn(...)`. Required by the signature. In this version the update of $\tilde B$ is the adaptive random-walk Metropolis, so these settings are not used. |
| `iter_start_zeta` | `Int64` (default `0`) | The update of *Z* starts after this many iterations. |
| `iter_start_hmc` | `Int64` (default `100`) | Currently not used. |
| `missing_index` | `Matrix{Int64}` (n_miss × 2) | Positions of the missing values (see above). The default is no missing values. |

### Output

A `NamedTuple`; the first dimension of every array indexes the saved draws (`S`).

| Field | Size | Content |
|---|---|---|
| `lambda_out` | S × k × k | Precision matrix Λ, in the **original** variable order. |
| `zmat_out` | S × k × k | Graph, original order. The strict lower triangle holds z_jk (1 = edge, 0 = zero); the diagonal and upper triangle are set to 1. |
| `beta_out` | S × q | Regression coefficients. |
| `missing_out` | S × n_miss | Imputed missing values, in the row order of `missing_index`. |
| `order_matrix_out` | S × k × k | Permutation matrix *D* at each saved draw. |
| `q_mat_sparse_out` | S × k × k | Cholesky factor *Q* (**permuted** order, i.e. of $D\Lambda D^\top$). |
| `m_out`, `c_out` | S × k(k-1)/2, S × k | Off-diagonal and diagonal entries of $\tilde B$ (permuted order). |

---

## Current limitations

This release is not finished. Known limitations:

- The scale matrix must be **S = ψ²I**. A non-diagonal *S*, or a diagonal *S* with unequal
  entries, raises an error.
- **k ≥ 10 is currently required.** A leftover debug print in `mcmc` indexes the first 10×10
  block of *S*. It will be removed.
- `mcmc_glm` (non-Gaussian responses, e.g. Poisson, through a latent Gaussian layer) is
  **experimental and does not run** in this version.
- `GenCorSparse_V2` (a variant with separately modelled marginal variances), the HMC/NUTS
  samplers in `src/hmc.jl`, and the other exported types are experimental.
- There is no test suite yet, and the documentation in `docs/` is not built.

---

## Reference

If you use this package, please cite:

> G. Mastrantonio, P. Alaimo Di Loro, M. Mingione.
> *A new hierarchical distribution on arbitrary sparse precision matrices.* Working paper.

## License

MIT. See [LICENSE](LICENSE).
