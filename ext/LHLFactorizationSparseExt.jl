"""
    LHLFactorizationSparseExt

Sparse LHL: solve a family of shifted systems `(σI + τJ) x = b` for a **sparse** `J`.  A
Hessenberg similarity fills in an irreducible sparse block, so the only sparsity the LHL
reduction can exploit is *reducibility* — the block triangular form KLU is built around.
This extension therefore:

 1. takes the symmetric block triangular form of `J` (strongly connected components of its
    digraph; a `σ ≠ 0` puts a structural diagonal on every row, so the BTF of `σI + τJ` is
    exactly the SCC decomposition, and — unlike KLU's — the permutation is symmetric, which a
    similarity needs),
 2. reduces each irreducible diagonal block once with the dense LHL kernels of this package
    (packed small-block storage, or an [`LHLWorkspace`](@ref) above `small_max`), or — where a
    cost model finds it cheaper — factors it with a sparse LU ([PureKLU](https://github.com/SciML/PureKLU.jl))
    refactored per shift,
 3. keeps the off-diagonal blocks sparse and untransformed, applied in the block
    back-substitution.

It reuses the package's verbs on a sparse `J`:

```julia
using LHLFactorization, SparseArrays, PureKLU
F = lhl(J)                        # analyze + reduce (J::SparseMatrixCSC)
for γ in (0.01, 0.013, 0.021)
    lhl_shift!(F, 1, -γ)          # load I - γJ
    x = lhl_ldiv!(copy(b), F)     # solve (also ldiv!, F \\ b, a matrix right-hand side)
    lhl_refine!(x, I - γ*J, b, F, 1)
end
lhl!(F, J2)                       # re-reduce with new values, same pattern
```

`lhl(J; shift = Complex{eltype(J)})` keeps the reduction real and the shifts/solves complex
(the Radau case); `lhl(J; thread = true)` (with `julia -t N`) threads the shift and the
reduction over the diagonal blocks; a `Complex` or matrix right-hand side on a real
factorization is solved by channels.  See the keyword arguments of [`lhl`](@ref).
"""
module LHLFactorizationSparseExt

using LHLFactorization: LHLFactorization, LHLWorkspace, LHLShift, lhl_reduce!, lhl_shift!, lhl_ldiv!
import LHLFactorization: lhl, lhl!, lhl_shift!, lhl_ldiv!, lhl_refine!, lhl_isreduced
using LinearAlgebra: LinearAlgebra, checksquare, mul!, ldiv!
using SparseArrays: SparseArrays, SparseMatrixCSC, nnz
using PureKLU: PureKLU

# The tested sparse implementation (SparseLHL): SCC ordering, packed small-block kernels,
# symbolic analysis + cost model, the reduce/shift/solve/refine driver, and the multi-RHS
# kernels.  Kept verbatim; the public entry points are bridged to the package verbs below.
include("sparse/scc.jl")
include("sparse/smallblock.jl")
include("sparse/symbolic.jl")
include("sparse/numeric.jl")
include("sparse/multirhs.jl")

# ---------------------------------------------------------------------------
# Bridge the sparse solver onto the package's dense verbs.  A sparse `J` dispatches `lhl`
# to the analysis-and-reduction here; the returned `SparseLHLFactorization` then answers the
# same `lhl!` / `lhl_shift!` / `lhl_ldiv!` / `lhl_refine!` as an `LHLWorkspace`.
# ---------------------------------------------------------------------------

"""
    lhl(J::SparseMatrixCSC; shift = eltype(J), kernel = :auto, small_max = 64, lu_min = 24,
        thread = false, kwargs...) -> SparseLHLFactorization

Analyze and reduce a square sparse `J` for the shifted family `(σI + τJ)`.  Follow with
[`lhl_shift!`](@ref) and [`lhl_ldiv!`](@ref).  `shift` is the element type of the shifts and
solves (`eltype(J)` or `Complex{real(eltype(J))}`); `kernel` is `:auto` (per-block cost
model), `:lhl` (dense LHL on every block) or `:lu` (PureKLU on every block); `thread = true`
threads the shift and reduction over blocks.  See the extension's docstring and
`slhl_analyze` for the full keyword set.
"""
lhl(J::SparseMatrixCSC; kwargs...) = slhl(J; kwargs...)

lhl!(F::SparseLHLFactorization, J::SparseMatrixCSC) = slhl!(F, J)

lhl_shift!(F::SparseLHLFactorization, σ, τ) = slhl_shift!(F, σ, τ)

lhl_ldiv!(x::AbstractVector, F::SparseLHLFactorization) = slhl_ldiv!(x, F)
lhl_ldiv!(X::AbstractMatrix, F::SparseLHLFactorization) = slhl_ldiv!(X, F)

# match the dense `lhl_refine!(x, A, b, ws, steps)` signature
lhl_refine!(x::AbstractVector, A, b::AbstractVector, F::SparseLHLFactorization, steps::Int) =
    slhl_refine!(x, A, b, F, steps)

lhl_isreduced(F::SparseLHLFactorization) = F.epoch[] > 0

end # module
