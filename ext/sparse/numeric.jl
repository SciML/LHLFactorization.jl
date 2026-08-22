# ---------------------------------------------------------------------------
# The factorization object: reduce, shift, solve, refine
# ---------------------------------------------------------------------------

"""
    SparseLHLFactorization

A sparse LHL factorization of `J` for the shifted family `σI + τJ`, built by [`slhl`](@ref).
Holds the symbolic analysis ([`SparseLHLSymbolic`](@ref)) and, per diagonal block of the
block triangular form, either a dense LHL reduction (on packed small-block storage or in an
`LHLFactorization.LHLWorkspace`), a sparse LU (PureKLU) that is refactored per shift, or a
scalar.  The off-diagonal blocks are kept sparse and untransformed: the block
back-substitution of the solve uses them as they are, so only the diagonal blocks ever pay
for the similarity transformation.

Lifecycle: [`slhl!`](@ref) re-reduces with new values on the same pattern (`O(Σ b³)` over
the dense blocks), [`slhl_shift!`](@ref) loads a shift (`O(Σ b²)` plus the LU blocks'
refactorizations), [`slhl_ldiv!`](@ref) / `ldiv!` solves, [`slhl_refine!`](@ref) refines.
`F.info` is the (permuted) index of the first zero pivot of the current shift, `0` if none;
`F.σ`, `F.τ` the shift loaded.  A factorization owns scratch buffers, so it must not serve
concurrent solves.

`SparseLHLFactorization{T, TG}`: `T` is the matrix's element type — the element type of the
`O(b³)` reductions — and `TG` that of the shifts and solves, `T` or `Complex{real(T)}`
(`slhl(J; shift = ComplexF64)` on a real `J`: the reductions stay real, the `O(b²)` shifted
Hessenberg LUs, the LU blocks' refactorizations and the solves are complex — the Radau case,
as LHLFactorization's `LHLShift{Complex{T}}`).  The shift-dependent storage (`gstore`,
`sinv`, `luwork`, `X`, `resid`, the channel scratches) has element type `TG`.

    SparseLHLFactorization(F; shift = ...)

builds a second factorization that **shares `F`'s reduction** (its analysis, the dense
blocks' reductions, the LU blocks' values) and holds its own shift state — so one real
reduction serves a real and a complex shift at the same time, each with its own
`slhl_shift!`/`slhl_ldiv!`.  A later `slhl!` through either object re-reduces for both.
"""
mutable struct SparseLHLFactorization{T, TG, WS, SH, KF} <: LinearAlgebra.Factorization{T}
    # `J` is held with `Int` indices whatever the user's index type (an `Int32` matrix has
    # its two index arrays converted once, its values shared), so that the index type is not
    # a type parameter and Int32 and Int64 matrices share one compiled kernel set; `Jref` is
    # the user's object itself, so `slhl!(F, J)` with the same object skips the conversion.
    J::SparseMatrixCSC{T, Int}
    Jref::Any
    sym::SparseLHLSymbolic
    vstore::Vector{T}          # small blocks: factors, Hᵀ, balancing (the reductions)
    gstore::Vector{TG}         # small blocks: Gᵀ, pivot reciprocals (the current shift)
    istore::Vector{Int}
    swap::Vector{Bool}
    sdiag::Vector{T}
    sinv::Vector{TG}
    ws::Vector{WS}             # large blocks' reductions (LHLWorkspace, shared between factorizations)
    lhsh::Vector{SH}           # large blocks' shift state (LHLShift{TG}, one per factorization)
    bigA::Matrix{T}
    lu::Vector{KF}
    luvals::Vector{Vector{T}}
    luwork::Vector{Vector{TG}}
    lufactored::Vector{Bool}
    offx::Vector{T}
    X::Vector{TG}
    resid::Vector{TG}
    swork::Vector{TG}
    solve_scratch::Matrix{TG}   # n × m channels of a mixed-element-type solve, grown on demand
    refine_scratch::Matrix{TG}  # the right-hand side's channels in a mixed-element-type refinement
    mwork::Vector{TG}           # the multi-RHS driver's dense-block work area (4·largest block)
    perm_scratch::Matrix{TG}    # a matrix / refinement right-hand side's channels, permuted order
    thread::Bool                # thread the shift and the reduction over blocks (Base threads)
    sworks::Vector{Vector{TG}}  # per-thread pending-row scratch for a threaded shift
    bigAs::Vector{Matrix{T}}    # per-thread dense staging for a threaded reduction
    shift_chunks::Vector{Int}   # block-chunk boundaries for the threaded shift (empty: serial)
    reduce_chunks::Vector{Int}  # block-chunk boundaries for the threaded reduction
    σ::TG
    τ::TG
    info::Int
    epoch::Base.RefValue{Int}  # the reduction's generation, shared by factorizations on one reduction (0: none)
    luepoch::Int               # the generation the LU blocks' pivot sequences belong to
    shifted::Bool
end
_isreduced(F::SparseLHLFactorization) = F.epoch[] > 0

# The same matrix with `Int` index arrays: `J` itself when it already has them, otherwise a
# wrapper sharing `J`'s values (the two index arrays are converted).
_with_int_indices(J::SparseMatrixCSC{T, Int}) where {T} = J
_with_int_indices(J::SparseMatrixCSC{T}) where {T} =
    SparseMatrixCSC(size(J, 1), size(J, 2), convert(Vector{Int}, J.colptr), convert(Vector{Int}, J.rowval), J.nzval)

_wstype(::Type{T}, ::Type{TG}) where {T, TG} = LHLWorkspace{T, real(T), TG}
_shtype(::Type{TG}) where {TG} = LHLShift{TG, real(TG)}
_kftype(::Type{TG}) where {TG} = PureKLU.KLUFactorization{TG, Int, real(TG)}

# The shift-dependent half for the LU blocks of a factorization: a PureKLU factorization per
# block (of the shift's element type), its numeric sized by PureKLU's allocator so that the
# first real factorization allocates nothing.
function _lu_blocks(::Type{TG}, sym::SparseLHLSymbolic) where {TG}
    KF = _kftype(TG)
    lu = Vector{KF}(undef, sym.nlu)
    luwork = Vector{Vector{TG}}(undef, sym.nlu)
    @inbounds for k in 1:sym.nblocks
        sym.kind[k] == KIND_LU || continue
        b = sym.R[k + 1] - sym.R[k]
        l = sym.kidx[k]
        nz = length(sym.lurowval[l])
        w = zeros(TG, nz)
        luwork[l] = w
        S = SparseMatrixCSC{TG, Int}(b, b, sym.lucolptr[l], sym.lurowval[l], w)
        prealloc = sym.luprealloc < 0 ? nothing : sym.luprealloc > 0
        K = PureKLU.klu(S; full_factor = false, tol = sym.lutol, fully_preallocated = prealloc)
        K.common.scale = Cint(sym.luscale)
        setfield!(K, :numeric, PureKLU._alloc_numeric(TG, getfield(K, :symbolic), K.common, K.colptr))
        lu[l] = K
    end
    return lu, luwork
end

"""
    SparseLHLFactorization(J, sym; shift = eltype(J))

Allocate the factorization for `J` over the symbolic analysis `sym` (no reduction yet —
[`slhl!`](@ref) does that); `shift` is the element type of the shifts and solves, `eltype(J)`
or `Complex{real(eltype(J))}`.
"""
function SparseLHLFactorization(J::SparseMatrixCSC{T}, sym::SparseLHLSymbolic; shift::Type{TG} = T, thread::Bool = false) where {T, TG}
    (TG === T || TG === Complex{real(T)}) ||
        throw(ArgumentError("shift must be $T or $(Complex{real(T)}), got $TG"))
    n = sym.n
    WS = _wstype(T, TG)
    SH = _shtype(TG)
    ws = Vector{WS}(undef, sym.nbig)
    lhsh = Vector{SH}(undef, sym.nbig)
    luvals = Vector{Vector{T}}(undef, sym.nlu)
    maxsmall = 1
    @inbounds for k in 1:sym.nblocks
        b = sym.R[k + 1] - sym.R[k]
        kd = sym.kind[k]
        if kd == KIND_SMALL
            maxsmall = max(maxsmall, b)
        elseif kd == KIND_BIG
            g = sym.kidx[k]
            ws[g] = LHLWorkspace{T}(b; shift = TG)
            lhsh[g] = LHLShift{TG}(ws[g])
        elseif kd == KIND_LU
            luvals[sym.kidx[k]] = zeros(T, length(sym.lurowval[sym.kidx[k]]))
        end
    end
    lu, luwork = _lu_blocks(TG, sym)
    KF = eltype(lu)
    nt = Threads.nthreads()
    sworks = [zeros(TG, maxsmall + 4) for _ in 1:(thread ? nt : 1)]
    mb = sym.nbig > 0 ? sym.maxbig : 0
    bigAs = [Matrix{T}(undef, mb, mb) for _ in 1:(thread ? nt : 1)]
    schunks = thread ? _block_chunks(sym, _shift_cost, nt) : Int[]
    rchunks = thread ? _block_chunks(sym, _reduce_cost, nt) : Int[]
    return SparseLHLFactorization{T, TG, WS, SH, KF}(
        _with_int_indices(J), J, sym, zeros(T, sym.vlen), zeros(TG, sym.glen), zeros(Int, sym.ilen),
        zeros(Bool, sym.slen), zeros(T, sym.nscalar), zeros(TG, sym.nscalar), ws, lhsh,
        bigAs[1], lu, luvals, luwork, zeros(Bool, sym.nlu),
        zeros(T, length(sym.offmap)), zeros(TG, n), zeros(TG, n), zeros(TG, maxsmall + 4),
        Matrix{TG}(undef, 0, 0), Matrix{TG}(undef, 0, 0),
        zeros(TG, 4 * max(maxsmall, sym.maxbig) + 8), Matrix{TG}(undef, 0, 0),
        thread, sworks, bigAs, schunks, rchunks, zero(TG), zero(TG), 0, Ref(0), 0, false
    )
end

"""
    SparseLHLFactorization(F::SparseLHLFactorization; shift = ...)

A second factorization on `F`'s reduction: it shares the analysis, the dense blocks'
reductions and the LU blocks' values, and owns its shift state (shifted Hessenberg LUs, LU
refactorizations, scratch) in the element type `shift` — `eltype(J)` or
`Complex{real(eltype(J))}`.  One real reduction can so serve a real and a complex shift at
once, each object with its own `slhl_shift!` / `slhl_ldiv!` / `slhl_refine!` (the Radau
case).  `slhl!` through either object re-reduces for both; the two must not be used from
concurrent threads (they share the reduction's scratch).
"""
function SparseLHLFactorization(F::SparseLHLFactorization{T}; shift::Type{TG} = T) where {T, TG}
    (TG === T || TG === Complex{real(T)}) ||
        throw(ArgumentError("shift must be $T or $(Complex{real(T)}), got $TG"))
    sym = F.sym
    n = sym.n
    SH = _shtype(TG)
    lhsh = Vector{SH}(undef, sym.nbig)
    @inbounds for g in 1:sym.nbig
        lhsh[g] = LHLShift{TG}(F.ws[g])
    end
    lu, luwork = _lu_blocks(TG, sym)
    KF = eltype(lu)
    WS = eltype(F.ws)
    nt = Threads.nthreads()
    ms = length(F.swork) - 4
    sworks = [zeros(TG, ms + 4) for _ in 1:(F.thread ? nt : 1)]
    return SparseLHLFactorization{T, TG, WS, SH, KF}(
        F.J, F.Jref, sym, F.vstore, zeros(TG, sym.glen), F.istore,
        zeros(Bool, sym.slen), F.sdiag, zeros(TG, sym.nscalar), F.ws, lhsh,
        F.bigA, lu, F.luvals, luwork, zeros(Bool, sym.nlu),
        F.offx, zeros(TG, n), zeros(TG, n), zeros(TG, length(F.swork)),
        Matrix{TG}(undef, 0, 0), Matrix{TG}(undef, 0, 0),
        zeros(TG, length(F.mwork)), Matrix{TG}(undef, 0, 0),
        F.thread, sworks, F.bigAs, copy(F.shift_chunks), copy(F.reduce_chunks), zero(TG), zero(TG), 0, F.epoch, 0, false
    )
end

Base.size(F::SparseLHLFactorization) = (F.sym.n, F.sym.n)
Base.size(F::SparseLHLFactorization, d::Integer) = d <= 2 ? F.sym.n : 1
LinearAlgebra.issuccess(F::SparseLHLFactorization) = F.shifted && F.info == 0

Base.show(io::IO, F::SparseLHLFactorization) = show(io, MIME"text/plain"(), F)
function Base.show(io::IO, ::MIME"text/plain", F::SparseLHLFactorization{T, TG}) where {T, TG}
    s = F.sym
    print(
        io, "SparseLHLFactorization{$T}", TG === T ? "" : " (shift $TG)", ": $(s.n)×$(s.n), $(s.nblocks) blocks (",
        "$(s.nscalar) scalar, $(s.nsmall) small LHL, $(s.nbig) large LHL, $(s.nlu) sparse LU), ",
        "max block $(s.nblocks == 0 ? 0 : maximum(diff(s.R))), nnz(offdiag) = $(length(s.offi))"
    )
    F.shifted && print(io, "; shift (σ, τ) = ($(F.σ), $(F.τ)), info = $(F.info)")
    return nothing
end

"""
    slhl(J; shift = eltype(J), kwargs...) -> SparseLHLFactorization

Analyze ([`slhl_analyze`](@ref), whose keywords this takes) and reduce a square sparse `J`
for the shifted family `σI + τJ`.  Follow with [`slhl_shift!`](@ref) and
[`slhl_ldiv!`](@ref).  `shift` is the element type of the shifts and solves: `eltype(J)`, or
`Complex{real(eltype(J))}` for complex shifts on a real reduction (the `O(b³)` reductions
stay real; the per-shift work and the solves are complex).  `thread = true` (with
`julia -t N`) threads the shift and the reduction over the diagonal blocks with Base
threads, deterministically (bit-identical for any thread count) and only above a work
threshold; the solve stays serial.  `J` is not modified; the
factorization keeps a reference to it (with `Int` indices — an `Int32` matrix has its index
arrays converted once, its values shared) for [`slhl_refine!`](@ref).
"""
function slhl(J::SparseMatrixCSC{T}; shift::Type{TG} = T, thread::Bool = false, kwargs...) where {T, TG}
    sym = slhl_analyze(J; shift = TG, kwargs...)
    F = SparseLHLFactorization(J, sym; shift = TG, thread)
    return slhl_reduce!(F, J)
end

"""
    slhl!(F, J) -> F

Re-reduce with the values of `J`, which must have the sparsity pattern `F` was analyzed
for.  `O(Σ b³)` over the dense blocks; the LU blocks are refactored at the next shift.
"""
function slhl!(F::SparseLHLFactorization, J::SparseMatrixCSC)
    return slhl_reduce!(F, J)
end

function slhl_reduce!(F::SparseLHLFactorization{T}, J::SparseMatrixCSC) where {T}
    sym = F.sym
    n = sym.n
    size(J) == (n, n) || throw(DimensionMismatch("J is $(size(J)), the factorization $(n)×$(n)"))
    if J !== F.Jref
        (J.colptr == F.J.colptr && J.rowval == F.J.rowval) ||
            throw(ArgumentError("J's sparsity pattern differs from the one analyzed"))
        F.J = _with_int_indices(J)
        F.Jref = J
    end
    nz = F.J.nzval
    vs = F.vstore
    @inbounds begin
        offmap = sym.offmap
        offx = F.offx
        for p in eachindex(offmap)
            offx[p] = nz[offmap[p]]
        end
        for s in 1:sym.nscalar
            q = sym.sdiag[s]
            F.sdiag[s] = q == 0 ? zero(T) : nz[q]
        end
        if !isempty(F.reduce_chunks)
            _run_chunks(F.reduce_chunks) do k, t
                _reduce_block!(F, k, nz, F.bigAs[t])
                return 0
            end
        else
            for k in 1:sym.nblocks
                _reduce_block!(F, k, nz, F.bigA)
            end
        end
    end
    # a new generation of the reduction: every factorization on it re-pivots its LU blocks
    # at its next shift (see slhl_shift!), and this one's current shift is void
    F.epoch[] += 1
    F.shifted = false
    return F
end

# Gather the block's values and reduce it (small: packed LHL; big: dense LHL in its
# workspace, staged through `bigA`; LU: copy the values for the next shift's refactorization).
@inline function _reduce_block!(F::SparseLHLFactorization{T}, k::Int, nz::Vector{T}, bigA::Matrix{T}) where {T}
    sym = F.sym
    kd = sym.kind[k]
    kd == KIND_SCALAR && return nothing
    b = sym.R[k + 1] - sym.R[k]
    vs = F.vstore
    @inbounds if kd == KIND_SMALL
        s = sym.kidx[k]
        o = sym.voff[s]
        for i in 1:(b * b)
            vs[o + i] = zero(T)
        end
        d = sym.didx[k]
        for q in sym.bmapptr[d]:(sym.bmapptr[d + 1] - 1)
            vs[o + sym.bpos[q]] += nz[sym.bmap[q]]
        end
        _sb_balance!(vs, o, b)
        _sb_reduce!(vs, o, b, F.istore, sym.ioff[s])
    elseif kd == KIND_BIG
        A = view(bigA, 1:b, 1:b)
        fill!(A, zero(T))
        d = sym.didx[k]
        for q in sym.bmapptr[d]:(sym.bmapptr[d + 1] - 1)
            A[sym.bpos[q]] += nz[sym.bmap[q]]
        end
        lhl_reduce!(F.ws[sym.kidx[k]], A, true, Val(false))
    else
        l = sym.kidx[k]
        map = sym.lumap[l]
        base = F.luvals[l]
        for p in eachindex(map)
            q = map[p]
            base[p] = q == 0 ? zero(T) : nz[q]
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Threading of the shift and the reduction over the diagonal blocks (opt-in, `thread = true`,
# and `julia -t N`).  The blocks are independent — a shift or a reduction writes only its own
# block's storage — so a cost-balanced contiguous partition, one chunk per thread with its own
# scratch, is deterministic: the partition depends on the block sizes only, so `factors`, the
# shift and the solves are bit-identical for any thread count.  The solve is not threaded (its
# blocks depend through the off-diagonal part, and the block DAG of typical matrices is a chain
# or has levels too cheap to spawn for — measured).  Threading pays only above a work
# threshold (spawn/join is ~15 µs); below it the serial path runs.
_reduce_cost(sym, k) = _cost(sym, k, 3)
_shift_cost(sym, k) = _cost(sym, k, 1)
function _cost(sym::SparseLHLSymbolic, k::Int, which::Int)
    kd = sym.kind[k]
    kd == KIND_SCALAR && return 2.0
    b = Float64(sym.R[k + 1] - sym.R[k])
    if kd == KIND_LU
        e = sym.est[k]
        nnzb = Float64(length(sym.lurowval[sym.kidx[k]]))
        which == 3 && return 1.5 * nnzb
        fl = isnan(e.lu_flops) ? 3.0 * nnzb * sqrt(b) : e.lu_flops
        return 2.5 * fl
    end
    which == 3 && return 0.13 * b^3 + 2.0 * b^2       # reduce ≈ 5/3 b³
    return 0.27 * b^2 + 10.0 * b                      # shift ≈ b²
end

# Only thread when the total modelled work clears the spawn/join overhead (~15 µs) with
# margin: shift is `O(b²)` and cheaper, so it needs a bigger block set than the reduction.
const _THREAD_SHIFT_MIN_NS = 1.5e5
const _THREAD_REDUCE_MIN_NS = 1.5e5

# Contiguous partition of blocks 1:nblocks by cumulative cost into ≤ nt chunks (computed
# once, at construction, since it depends only on the analysis and the thread count); an
# empty result means run serially.  Determinism: the partition is a function of the block
# sizes only.
function _block_chunks(sym::SparseLHLSymbolic, costfn, nt::Int)
    (nt > 1 && sym.nblocks > 1) || return Int[]
    nb = sym.nblocks
    total = 0.0
    @inbounds for k in 1:nb
        total += costfn(sym, k)
    end
    total > (costfn === _shift_cost ? _THREAD_SHIFT_MIN_NS : _THREAD_REDUCE_MIN_NS) || return Int[]
    nt = min(nt, nb)
    bounds = Vector{Int}(undef, nt + 1)
    bounds[1] = 1
    acc = 0.0
    t = 1
    @inbounds for k in 1:nb
        acc += costfn(sym, k)
        while t < nt && acc >= total * t / nt
            t += 1
            bounds[t] = k + 1
        end
    end
    @inbounds for tt in (t + 1):nt
        bounds[tt] = nb + 1
    end
    bounds[nt + 1] = nb + 1
    j = 1
    @inbounds for i in 2:(nt + 1)
        if bounds[i] > bounds[j]
            j += 1
            bounds[j] = bounds[i]
        end
    end
    j >= 3 || return Int[]        # < 2 real chunks: run serially (a lone spawn is pure overhead)
    return resize!(bounds, j)
end

# Run `f(k, t)` over every block, `t` the 1-based chunk (thread-slot) index; returns the
# minimum positive return value (0 = none) — the first zero pivot in block order.
function _run_chunks(f::Fn, bounds::Vector{Int}) where {Fn}
    nchunk = length(bounds) - 1
    infos = fill(0, nchunk)
    @sync for t in 1:nchunk
        Threads.@spawn begin
            local best = 0
            @inbounds for k in bounds[t]:(bounds[t + 1] - 1)
                ik = f(k, t)
                ik != 0 && best == 0 && (best = ik)
            end
            infos[t] = best
        end
    end
    info = 0
    @inbounds for t in 1:nchunk
        infos[t] != 0 && (info == 0 || infos[t] < info) && (info = infos[t])
    end
    return info
end

# One block's contribution to the shift; returns the position of its first zero pivot (0 if
# none).  Small blocks use the per-thread pending-row scratch `swork`.
@inline function _shift_block!(F::SparseLHLFactorization{T, TG}, k::Int, σ::TG, τ::TG, swork::Vector{TG}) where {T, TG}
    sym = F.sym
    R = sym.R
    kd = sym.kind[k]
    @inbounds if kd == KIND_SCALAR
        return iszero(σ + τ * F.sdiag[sym.kidx[k]]) ? R[k] : 0
    elseif kd == KIND_SMALL
        i = sym.kidx[k]
        ik = _sb_shift_dispatch!(F.vstore, sym.voff[i], F.gstore, sym.goff[i], R[k + 1] - R[k], F.swap, sym.soff[i], swork, σ, τ)
        return ik == 0 ? 0 : R[k] + ik - 1
    elseif kd == KIND_BIG
        g = sym.kidx[k]
        sh = F.lhsh[g]
        lhl_shift!(sh, F.ws[g], σ, τ)
        return sh.info == 0 ? 0 : R[k] + sh.info - 1
    else
        ik = _lu_shift!(F, sym.kidx[k], σ, τ)
        return ik == 0 ? 0 : R[k] + ik - 1
    end
end

"""
    slhl_shift!(F, σ, τ) -> F

Load the shift `σI + τJ`: the Hessenberg LU of every dense block (`O(b²)` each), the
refactorization of every sparse-LU block, the scalar blocks' reciprocals.  `(1, -γ)` gives
`I - γJ`.  A zero pivot is reported in `F.info`, not thrown.
"""
function slhl_shift!(F::SparseLHLFactorization{T, TG}, σ, τ) where {T, TG}
    _isreduced(F) || throw(ArgumentError("the factorization holds no reduction; call slhl! first"))
    if F.luepoch != F.epoch[]
        fill!(F.lufactored, false)     # the LU blocks' values changed: pivot afresh
        F.luepoch = F.epoch[]
    end
    if TG <: Real && !(isreal(σ) && isreal(τ))
        throw(
            ArgumentError(
                "complex shift ($σ, $τ) on a factorization with real shifts; build it with " *
                    "`slhl(J; shift = $(Complex{real(T)}))`"
            )
        )
    end
    σ = convert(TG, σ)
    τ = convert(TG, τ)
    F.σ = σ
    F.τ = τ
    sym = F.sym
    swork = F.swork
    sdiag = F.sdiag
    sinv = F.sinv
    # the scalar blocks as one contiguous pass: the divides vectorize
    @inbounds @simd for i in 1:sym.nscalar
        sinv[i] = inv(σ + τ * sdiag[i])
    end
    if !isempty(F.shift_chunks)
        info = _run_chunks(F.shift_chunks) do k, t
            _shift_block!(F, k, σ, τ, F.sworks[t])
        end
    else
        info = 0
        @inbounds for k in 1:sym.nblocks
            ik = _shift_block!(F, k, σ, τ, swork)
            ik != 0 && info == 0 && (info = ik)
        end
    end
    F.info = info
    F.shifted = true
    return F
end

# Values τ·J + σ·I on the block, then PureKLU's refactorization with the pivot sequence of
# the first factorization; a refactorization that meets a zero pivot re-pivots.
function _lu_shift!(F::SparseLHLFactorization{T, TG}, l::Int, σ::TG, τ::TG) where {T, TG}
    sym = F.sym
    base = F.luvals[l]
    w = F.luwork[l]
    K = F.lu[l]
    dg = sym.ludiag[l]
    @inbounds @simd for p in eachindex(w)
        w[p] = τ * base[p]
    end
    @inbounds for j in eachindex(dg)
        w[dg[j]] += σ
    end
    if F.lufactored[l]
        PureKLU.klu!(K, w; check = false, allowsingular = true)
        if K.common.status == PureKLU.KLU_SINGULAR
            K.nzval = w
            PureKLU.klu_factor!(K; check = false, allowsingular = true)
        end
    else
        K.nzval = w
        PureKLU.klu_factor!(K; check = false, allowsingular = true)
    end
    status = K.common.status
    status >= PureKLU.KLU_OK ||
        error("PureKLU failed on a diagonal block (status code $(Int(status))); the block's pattern is invalid")
    # Only a healthy factorization's pivot sequence is reused by the next shift's fixed-pivot
    # refactorization: not a singular one (a rank-deficient L/U pattern, which PureKLU's klu!
    # would happily refactor and report OK), and not one chosen on a numerically singular
    # matrix (tiny pivots, status OK) — min|U_ii| / max|U_ii| below `lurepivot` re-pivots next
    # time as well.
    F.lufactored[l] = status == PureKLU.KLU_OK && _healthy_pivots(K, sym.lurepivot)
    if status == PureKLU.KLU_SINGULAR
        c = Int(K.common.singular_col)
        return c >= 0 ? c + 1 : 1
    end
    return 0
end

# min|U_ii| / max|U_ii| of a PureKLU numeric (the diagonal of U is stored separately), an
# O(b) proxy of KLU's rcond.
function _healthy_pivots(K::PureKLU.KLUFactorization{Tv}, tol::Float64) where {Tv}
    tol <= 0 && return true
    ud = getfield(K, :numeric).Udiag
    Tr = real(Tv)
    umin = typemax(Tr)
    umax = zero(Tr)
    @inbounds @simd for i in eachindex(ud)
        a = abs(ud[i])
        umin = min(umin, a)
        umax = max(umax, a)
    end
    return umin > tol * umax
end

# The diagonal block k's solve on its vector `xv` (the block's slice of the solve vector,
# or the vector itself when the matrix is one block), by its kernel.
@inline function _block_solve!(F::SparseLHLFactorization, k::Int, xv::AbstractVector)
    sym = F.sym
    kd = sym.kind[k]
    b = sym.R[k + 1] - sym.R[k]
    @inbounds if kd == KIND_SCALAR
        xv[1] *= F.sinv[sym.kidx[k]]
    elseif kd == KIND_SMALL
        s = sym.kidx[k]
        _sb_solve_dispatch!(F.vstore, sym.voff[s], F.gstore, sym.goff[s], b, F.istore, sym.ioff[s], F.swap, sym.soff[s], F.swork, xv, 0)
    elseif kd == KIND_BIG
        g = sym.kidx[k]
        lhl_ldiv!(xv, F.lhsh[g], F.ws[g])
    else
        PureKLU.solve!(F.lu[sym.kidx[k]], xv; check = false)
    end
    return nothing
end

"""
    slhl_ldiv!(x, F) -> x

`x ← (σI + τJ)⁻¹ x` for the shift loaded by [`slhl_shift!`](@ref): block back-substitution
over the block triangular form, each dense block through `Z⁻¹`, its Hessenberg solve and
`Z`, each LU block through its sparse triangular solves, the off-diagonal blocks as sparse
multiply–adds.  `LinearAlgebra.ldiv!(F, x)`, `ldiv!(y, F, x)` and `F \\ b` are the same.

`x` may have an element type other than the factorization's shift type `TG`: a
`Complex{TG}` (or narrower complex) right-hand side on a real factorization, a real or
complex type that promotes to `TG`, or — with ForwardDiff loaded — a `ForwardDiff.Dual` on
a real factorization.  The factorization is a linear operator, so such an `x` is solved by
channels: its `TG`-components are laid out as the columns of a reusable `n × m` scratch,
each column goes through the ordinary solve, and the components are reassembled
(PureKLU's real-factor / Dual-RHS path).  Allocation-free after the first call with a given
`m`; exact in `TG` arithmetic.
"""
function slhl_ldiv!(x::AbstractVector{S}, F::SparseLHLFactorization{T, TG}) where {S, T, TG}
    S === TG && return _ldiv_same!(x, F)
    m = _channels(S, TG)
    m == 0 && throw(ArgumentError(_mixed_error(S, TG)))
    return _ldiv_channels!(x, F, m)
end

function _mixed_error(::Type{S}, ::Type{T}) where {S, T}
    msg = "cannot solve in place a right-hand side of element type $S with a $T factorization"
    if S <: Real && T <: Complex
        msg *= ": a real x cannot hold the complex solution; use `F \\ b` or a complex x"
    elseif S <: Real && !(S <: AbstractFloat)
        msg *= " (a ForwardDiff.Dual right-hand side needs `using ForwardDiff`)"
    elseif promote_type(S, T) !== T
        msg *= ": $S does not fit the factorization's channels; use `F \\ b`, which promotes"
    end
    return msg
end

function _ldiv_same!(x::AbstractVector, F::SparseLHLFactorization{T, TG}) where {T, TG}
    F.shifted || throw(ArgumentError("no shift loaded; call slhl_shift! first"))
    sym = F.sym
    n = sym.n
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), the factorization is $(n)×$(n)"))
    # One block: the SCC order is the identity, the block kernel works on x in place.
    if sym.nblocks == 1
        _block_solve!(F, 1, x)
        return x
    end
    X = F.X
    perm = sym.perm
    R = sym.R
    τ = F.τ
    offp = sym.offp
    offi = sym.offi
    offx = F.offx
    kind = sym.kind
    kidx = sym.kidx
    voff = sym.voff
    goff = sym.goff
    ioff = sym.ioff
    soff = sym.soff
    vstore = F.vstore
    gstore = F.gstore
    istore = F.istore
    swap = F.swap
    swork = F.swork
    sinv = F.sinv
    @inbounds for k in 1:n
        X[k] = x[perm[k]]
    end
    @inbounds for k in sym.nblocks:-1:1
        k1 = R[k]
        k2 = R[k + 1] - 1
        kd = kind[k]
        if kd == KIND_SCALAR
            X[k1] *= sinv[kidx[k]]
        elseif kd == KIND_SMALL
            i = kidx[k]
            _sb_solve_dispatch!(vstore, voff[i], gstore, goff[i], k2 - k1 + 1, istore, ioff[i], swap, soff[i], swork, X, k1 - 1)
        elseif kd == KIND_BIG
            g = kidx[k]
            lhl_ldiv!(view(X, k1:k2), F.lhsh[g], F.ws[g])
        else
            PureKLU.solve!(F.lu[kidx[k]], view(X, k1:k2); check = false)
        end
        # the block's columns update the rows of earlier blocks; offp is monotone, so one
        # comparison tells whether the block has any off-diagonal entry at all
        if k > 1 && offp[k2 + 1] > offp[k1]
            for c in k1:k2
                xc = τ * X[c]
                iszero(xc) && continue
                # the rows of one column are distinct (CSC) and all lie in earlier blocks
                @simd ivdep for p in offp[c]:(offp[c + 1] - 1)
                    X[offi[p]] -= offx[p] * xc
                end
            end
        end
    end
    @inbounds for k in 1:n
        x[perm[k]] = X[k]
    end
    return x
end

# ---------------------------------------------------------------------------
# Mixed element types.  `_channels(S, T)` is the number of `T`-columns a right-hand side of
# element type `S` occupies (0 = not solvable with this factorization); `_split_channels!`
# and `_merge_channels!` move the components.  The ForwardDiff extension adds `Dual`.
# ---------------------------------------------------------------------------
_channels(::Type{S}, ::Type{T}) where {S, T} = 0
_channels(::Type{S}, ::Type{T}) where {S <: Real, T <: Real} = promote_type(S, T) === T ? 1 : 0
_channels(::Type{S}, ::Type{T}) where {S <: Complex, T <: Complex} = promote_type(S, T) === T ? 1 : 0
_channels(::Type{Complex{S}}, ::Type{T}) where {S <: Real, T <: Real} = promote_type(S, T) === T ? 2 : 0

function _split_channels!(B::Matrix{T}, x::AbstractVector{S}, ::Type{S}) where {T, S}
    @inbounds for i in eachindex(x)
        B[i, 1] = x[i]
    end
    return B
end
function _merge_channels!(x::AbstractVector{S}, B::Matrix{T}, ::Type{S}) where {T, S}
    @inbounds for i in eachindex(x)
        x[i] = B[i, 1]
    end
    return x
end
function _split_channels!(B::Matrix{T}, x::AbstractVector{Complex{S}}, ::Type{Complex{S}}) where {T <: Real, S <: Real}
    @inbounds for i in eachindex(x)
        v = x[i]
        B[i, 1] = real(v)
        B[i, 2] = imag(v)
    end
    return B
end
function _merge_channels!(x::AbstractVector{Complex{S}}, B::Matrix{T}, ::Type{Complex{S}}) where {T <: Real, S <: Real}
    @inbounds for i in eachindex(x)
        x[i] = Complex{S}(B[i, 1], B[i, 2])
    end
    return x
end

# The channel scratches only ever grow: alternating channel counts (a complex solve, then a
# Dual one with a different chunk size) allocate nothing after their first use.
@inline function _grow_scratch!(F::SparseLHLFactorization{T, TG}, which::Val, n::Int, m::Int) where {T, TG}
    B = which === Val(:solve) ? F.solve_scratch : which === Val(:refine) ? F.refine_scratch : F.perm_scratch
    if size(B, 1) != n || size(B, 2) < m
        B = Matrix{TG}(undef, n, max(m, size(B, 1) == n ? size(B, 2) : 0))
        if which === Val(:solve)
            F.solve_scratch = B
        elseif which === Val(:refine)
            F.refine_scratch = B
        else
            F.perm_scratch = B
        end
    end
    return B
end

LinearAlgebra.ldiv!(F::SparseLHLFactorization, x::AbstractVector) = slhl_ldiv!(x, F)
function LinearAlgebra.ldiv!(y::AbstractVector, F::SparseLHLFactorization, x::AbstractVector)
    y === x || copyto!(y, x)
    return slhl_ldiv!(y, F)
end
# `F \ b` works on a copy whose element type is that of `b / F`'s entries, as for any
# `Factorization`: a `Float32` or `Int` `b` on a `Float64` factorization comes back `Float64`,
# a real `b` on a complex factorization complex, a `Complex{T}` or `Dual` `b` keeps its type
# (and solves through channels).
function _solve_copy(F::SparseLHLFactorization{T, TG}, B::AbstractVecOrMat{S}) where {T, TG, S}
    R = typeof(oneunit(S) / oneunit(TG))
    (R === TG || _channels(R, TG) != 0) ||
        throw(ArgumentError("cannot solve a right-hand side of element type $S with a $TG factorization"))
    return slhl_ldiv!(Array{R}(B), F)
end
Base.:\(F::SparseLHLFactorization, b::AbstractVector) = _solve_copy(F, b)
Base.:\(F::SparseLHLFactorization, B::AbstractMatrix) = _solve_copy(F, B)
# LinearAlgebra has `\(::Factorization{T}, ::VecOrMat{Complex{T}})` for BLAS reals; these are
# the more specific methods, so a complex right-hand side takes the channel solve.
Base.:\(F::SparseLHLFactorization{T}, b::Vector{Complex{T}}) where {T <: LinearAlgebra.BlasReal} = _solve_copy(F, b)
Base.:\(F::SparseLHLFactorization{T}, B::Matrix{Complex{T}}) where {T <: LinearAlgebra.BlasReal} = _solve_copy(F, B)
# ... and `\(::Factorization{T}, ::VecOrMat{T})` for the real `b` on a complex-shift F
Base.:\(F::SparseLHLFactorization{T, Complex{T}}, b::Vector{T}) where {T <: LinearAlgebra.BlasReal} = _solve_copy(F, b)
Base.:\(F::SparseLHLFactorization{T, Complex{T}}, B::Matrix{T}) where {T <: LinearAlgebra.BlasReal} = _solve_copy(F, B)

"""
    slhl_refine!(x, b, F, steps) -> x
    slhl_refine!(x, A, b, F, steps) -> x

`steps` rounds of fixed-precision iterative refinement of a solve of `(σI + τJ) x = b`,
each a residual (with the matrix `F` holds, or an explicit `A`) and a second solve.  The
similarity `Z` of a dense block is not orthogonal, so the raw solve's backward error carries
a factor `κ(Z)`; one step restores a backward error comparable to an LU's.
"""
function slhl_refine!(x::AbstractVector{S}, b::AbstractVector, F::SparseLHLFactorization{T, TG}, steps::Int) where {S, T, TG}
    steps <= 0 && return x
    S === TG && return _refine_same!(x, b, F, steps)
    # J, σ and τ are of the factorization's types, so the residual is linear over the
    # channels of a richer right-hand side: refine each channel on its own
    m = _channels(S, TG)
    (m == 0 || eltype(b) !== S) &&
        throw(ArgumentError("cannot refine a right-hand side of element type $S with a $TG factorization"))
    n = F.sym.n
    (length(x) == n && length(b) == n) || throw(DimensionMismatch("x and b must have length $n"))
    Bx = _grow_scratch!(F, Val(:solve), n, m)
    Bb = _grow_scratch!(F, Val(:refine), n, m)
    _split_channels!(Bx, x, S)
    _split_channels!(Bb, b, S)
    if m == 1
        _refine_same!(view(Bx, :, 1), view(Bb, :, 1), F, steps)
    else
        _refine_multi!(Bx, Bb, F, m, steps)
    end
    _merge_channels!(x, Bx, S)
    return x
end

function _refine_same!(x::AbstractVector, b::AbstractVector, F::SparseLHLFactorization, steps::Int)
    r = F.resid
    J = F.J
    σ = F.σ
    τ = F.τ
    for _ in 1:steps
        mul!(r, J, x)
        @inbounds @simd for i in eachindex(r)
            r[i] = b[i] - σ * x[i] - τ * r[i]
        end
        _ldiv_same!(r, F)
        x .+= r
    end
    return x
end

function slhl_refine!(x::AbstractVector, A, b::AbstractVector, F::SparseLHLFactorization, steps::Int)
    steps <= 0 && return x
    r = F.resid
    for _ in 1:steps
        mul!(r, A, x)
        r .= b .- r
        slhl_ldiv!(r, F)
        x .+= r
    end
    return x
end
