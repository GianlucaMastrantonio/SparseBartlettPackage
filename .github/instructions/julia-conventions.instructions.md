---
applyTo: "src/**/*.jl"
description: Enforce SparseBartlettPackage Julia coding conventions for all source files.
---

## Struct conventions

- Every struct **must** use a keyword-only inner constructor: `function MyStruct(; kwargs...)`.
- All fields must have explicit types — `Float64`, `Int64`, `Vector{Float64}`, `Matrix{Int64}`, etc. Never use `Any` or leave types implicit.
- Follow the existing pattern in `src/types.jl` (`order_lambda_V3`) as the canonical example.

## Function signatures

- All MCMC-facing functions use **keyword arguments only**.
- Iteration parameters are always passed as a `NamedTuple`:
  ```julia
  iterations::NamedTuple{(:iter, :burnin, :thin), Tuple{Int64, Int64, Int64}}
  ```
- Do not mix positional and keyword arguments in public API functions.

## Assertions and debug checks

- Use `@toggled_assert` (from `ToggleableAsserts`) instead of plain `@assert` on any hot path.
- Reserve plain `@assert` only for one-time setup validation outside the MCMC loop.

## Autodiff

- `ForwardDiff` and `Enzyme` are both active dependencies. Before adding gradient code, check which backend the target sampler already uses.
- `Zygote` is listed in `Project.toml` but all its call sites are commented out. Do **not** add new `Zygote`-based gradients without an explicit decision to re-introduce it.

## Module includes

- `sample_glm_z.jl` and `hmc_functions.jl` are **incomplete stubs** and are commented out in `SparseBartlettPackage.jl`. Do not re-enable them unless the implementation is complete.
- New source files must be added with an `include(joinpath("filename.jl"))` line in `SparseBartlettPackage.jl` and exported if they form part of the public API.
