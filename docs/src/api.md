# API Reference

## Module

```@docs
LHLFactorization
```

## Workspaces

```@docs
LHLWorkspace
LHLShift
```

## Reduction

```@docs
lhl
lhl!
lhl_reduce!
```

## Shifts and solves

```@docs
lhl_shift!
lhl_ldiv!
lhl_refine!
```

## Adjoint solves

```@docs
lhl_ldivH!
lhl_refineH!
```

## Similarity transformations

```@docs
applyZ!
applyZinv!
applyZH!
applyZinvH!
```

## Sparse routing

Predicates a consumer (for example a linear-solver stack) can use to decide when
the sparse `J` path — the `SparseArrays` + `PureKLU` extension — is worth taking.

```@docs
lhl_isreduced
lhl_prefers_sparse
```
