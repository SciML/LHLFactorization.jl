# Release Notes

## Unreleased

### Breaking

  - Removed `lhl_prefers_sparse`. Sparse solver selection belongs to the consuming solver
    stack; LHLFactorization's per-block cost model still selects between LHL and sparse LU
    kernels during factorization.

### Added

  - **Sparse `J`** (extension, loaded with `SparseArrays` and
    [PureKLU](https://github.com/SciML/PureKLU.jl)): `lhl(J::SparseMatrixCSC)` solves the
    shifted family `(σI + τJ)x = b` for a sparse Jacobian. A Hessenberg similarity fills in
    an irreducible sparse block, so the only sparsity the reduction can exploit is
    reducibility: the extension takes the symmetric block triangular form (strongly
    connected components — the BTF of `σI + τJ`, with a symmetric permutation a similarity
    needs), reduces each irreducible diagonal block with this package's dense kernels or,
    where a per-block cost model finds it cheaper, a PureKLU sparse LU refactored per shift,
    and keeps the off-diagonal blocks sparse. The same verbs work on the returned
    factorization (`lhl_shift!`, `lhl_ldiv!`, `lhl_refine!`, `lhl!`, `ldiv!`, `\`, a matrix
    right-hand side); `shift = Complex{eltype(J)}` gives complex shifts on a real reduction
    and `thread = true` threads the shift and reduction over blocks. It wins on
    block-triangular, dense-block or high-fill Jacobians and defers to a sparse LU
    elsewhere.
  - **Adjoint solves from the existing reduction**: `lhl_ldivH!` solves `Wᴴ x = b`
    against the same reduction and the same shift LU as `lhl_ldiv!`
    (`Wᴴ = Z⁻ᴴ Gᴴ Zᴴ` — three `O(n²)` phases, no refactorization), with `lhl_refineH!`
    for iterative refinement against `Aᴴ` and `applyZH!`/`applyZinvH!` for the adjoint
    similarity transformations. A complex shift on a real reduction conjugates only the
    shifted half: `Zᴴ = Zᵀ` stays real.
  - **Explicit-vector kernels for fully complex workspaces** (`lhl(J::Matrix{ComplexF64})`
    and `ComplexF32`). The reduction's trailing update, trailing GEMM and panel GEMV now
    run real explicit-vector kernels on the interleaved storage, and the solves' Z sweeps
    run on real planes (a planar copy of the packed multipliers). Measured on one Zen2
    core, a `ComplexF64` reduction runs at 2–3.8× the same-size real one (the flop ratio
    is 4) instead of 6.4–7.5×, and `lhl_ldiv!` at ≈2.5× instead of ≈4.7×. The threaded
    reduction now covers `ComplexF64` (`n ≥ 512`) and `ComplexF32` (`n ≥ 1024`), and is
    still bit-identical for any thread count. The `factors` layout is unchanged.
  - **Complex pivot magnitudes and balance norms are now `|re| + |im|`** (LAPACK's
    `CABS1`, as the shifted LU already used), replacing `abs`. Pivot *choices* of a
    complex reduction can therefore differ from v2.0.0 in near-ties; results differ at
    rounding level, both choices are equally valid partial pivoting, and complex
    multipliers are now bounded by `√2` in modulus (LAPACK's own bound for `zgetrf`)
    rather than 1. Real workspaces are bit-for-bit unaffected.

### Fixed

  - **`lhl_ldivH!` (and the `applyZinvH!` it drives) gave wrong `Float64` adjoint
    solves on 128-bit-SIMD targets** (e.g. AArch64/NEON, where a `Float64` vector holds
    two lanes). The packed back-substitution `_lhl_zinvsweepH_buf!` is pipelined: it
    issues the next group's body dot products before the current group's head is
    resolved, then folds the four intra-group coupling rows back in from registers. That
    fold treated the four rows as the first SIMD vector, which is only true when a vector
    spans at least four lanes; with two lanes the four rows straddle two vectors, so the
    second was counted twice and the solve was off by `O(1)`. The forward solve,
    `applyZH!`, the adjoint Hessenberg solve, and every `Float32` path (four lanes even
    at 128 bits) were already correct, as was every 256-bit target. The kernel now falls
    back to the width-agnostic generic sweep when a vector holds fewer than four lanes,
    and stays non-allocating. This landed before any release contained the adjoint API.

## v2.0

Breaking. The changes below are mechanical to adopt; `lhl`, `lhl!`, `lhl_reduce!`,
`applyZ!`/`applyZinv!` and the `LHLWorkspace{T}(n)` constructor are all source compatible
with v1.

### Breaking

  - **`LHLWorkspace` gained a third type parameter**, `LHLWorkspace{T, Tr}` →
    `LHLWorkspace{T, Tr, TG}`, where `TG` is the element type the shifted Hessenberg is
    held in. Only code that *writes the parameters out* is affected: an annotation
    `ws::LHLWorkspace{Float64, Float64}` must become `LHLWorkspace{Float64, Float64,
    Float64}` or simply `ws::LHLWorkspace`. The constructor `LHLWorkspace{T}(n)` is
    unchanged, so most code needs no edit at all.

  - **`lhl_ldiv!` now throws `DimensionMismatch`** when `x`'s length does not match the
    workspace. v1 read past the end and returned a wrong answer silently, so anything this
    now catches was already broken.

  - **The shift moved out of the workspace into `LHLShift`.** The shift-taking methods gain
    a shift argument — `lhl_shift!(sh, ws, σ, τ)`, `lhl_ldiv!(x, sh, ws)`,
    `lhl_refine!(x, A, b, sh, ws, steps)`. The one-shift forms `lhl_shift!(ws, σ, τ)`,
    `lhl_ldiv!(x, ws)` and `lhl_refine!(x, A, b, ws, steps)` still work and operate on the
    workspace's own shift, `ws.shift`; `ws.σ`, `ws.τ` and `ws.info` still read and write
    through to it. Code using one shift per reduction needs no change.

### Added

  - **`LHLShift{TG}`** — a shift is now a first-class object, so **one reduction can serve
    several shifts**. Build extra ones with `LHLShift{TG}(ws)`.

  - **Complex shifts on a real reduction**: `lhl(J; shift = ComplexF64)`, or
    `LHLWorkspace{T}(n; shift = TG)`. A real `J` keeps a real `O(n³)` reduction while the
    shift alone goes complex — the shape RadauIIA needs, where a real and a complex stage
    matrix come from one Jacobian.

  - **Threading**: `lhl(J; thread = Val(true))` (the default) and
    `lhl_reduce!(ws, J, balance, thread)` run the blocked reduction on
    [Polyester](https://github.com/JuliaSIMD/Polyester.jl) threads when Polyester is loaded
    and `Threads.nthreads() > 1`. Deterministic: the result is bit-identical for any thread
    count. Pass `Val(false)` to disable.

### Performance

  - Rewritten kernels. Measured against LAPACK on one thread, a re-shift is 14× cheaper
    than an LU refactorization at `n = 25` and 108× at `n = 800`, and `lhl_ldiv!` is now
    about the cost of an LU's triangular solves rather than twice it.

### Notes

  - **One workspace, and one `LHLShift`, cannot serve concurrent solves** — both write
    scratch buffers. Give each thread its own.
