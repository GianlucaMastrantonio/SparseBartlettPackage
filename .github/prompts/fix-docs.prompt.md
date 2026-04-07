---
mode: agent
description: Fix all stale SparseIW references in docs and README to use the correct SparseBartlettPackage name.
---

Fix all stale references to the old package name `SparseIW` / `SparseIW.jl` throughout the repository, replacing them with the correct name `SparseBartlettPackage` / `SparseBartlettPackage.jl`.

## Files to update

1. **`docs/make.jl`**
   - Replace `using SparseIW` → `using SparseBartlettPackage`
   - Replace `modules = [SparseIW]` → `modules = [SparseBartlettPackage]`
   - Replace `sitename = "SparseIW.jl"` → `sitename = "SparseBartlettPackage.jl"`

2. **`README.md`**
   - Update the `build` badge URL: replace `SparseIW.jl` with `SparseBartlettPackage` in both the image src and the href
   - Update the `codecov.io` badge URLs similarly
   - Update any commented-out Documentation badge URLs

3. **`docs/src/index.md`**
   - Update the title heading if it still says `SparseIW`

## After editing

Run a final grep to confirm no remaining `SparseIW` occurrences exist in the repository:

```
grep -r "SparseIW" .
```

Report any remaining occurrences that could not be safely auto-fixed.
