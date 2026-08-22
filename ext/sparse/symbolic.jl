# ---------------------------------------------------------------------------
# Symbolic analysis: block triangular form, per-block kernel choice, gather maps
# ---------------------------------------------------------------------------

const KIND_SCALAR = 0x00   # 1×1 block: a scalar divide
const KIND_SMALL = 0x01    # dense LHL on the flat small-block storage
const KIND_BIG = 0x02      # dense LHL in an LHLFactorization workspace
const KIND_LU = 0x03       # sparse LU (PureKLU) refactored at every shift

"""
    CostModel(; kwargs...)

Time model of the two per-block kernels, in nanoseconds, used by `kernel = :auto` of
[`slhl_analyze`](@ref).  For a dense LHL block of size `b`:

    shift   = lhl_shift_per_b2·b² + lhl_shift_per_b·b
    solve   = lhl_solve_per_b2·b² + lhl_solve_per_b·b
    reduce  = lhl_reduce_per_b3·b³ + lhl_reduce_per_b2·b²

and for a sparse LU block with `flops` multiply–adds per left-looking refactorization and
`nnz` = nnz(L+U):

    refactor = lu_refactor_per_flop·flops + lu_refactor_per_col·b
    solve    = lu_solve_per_nnz·nnz + lu_solve_per_row·b
    setup    = lu_setup_factor·refactor + lu_setup_per_col·b

The defaults were fitted on one core of an Apple M2 Max (Julia 1.11) against the packed
and workspace LHL kernels, PureKLU's refactorization and solve on dense blocks of 2–256 and
on the sparse matrices of the benchmark suite; they are ratios of throughputs more than
absolute times, so they transfer to other machines reasonably.  [`lhl_step_ns`](@ref) and
[`lu_step_ns`](@ref) evaluate the model.
"""
Base.@kwdef struct CostModel
    lhl_shift_per_b2::Float64 = 0.27
    lhl_shift_per_b::Float64 = 10.0
    lhl_solve_per_b2::Float64 = 0.3
    lhl_solve_per_b::Float64 = 12.0
    lhl_reduce_per_b3::Float64 = 0.13
    lhl_reduce_per_b2::Float64 = 2.0
    lu_refactor_per_flop::Float64 = 0.55
    lu_refactor_per_col::Float64 = 60.0
    lu_solve_per_nnz::Float64 = 0.55
    lu_solve_per_row::Float64 = 8.0
    lu_setup_factor::Float64 = 2.0
    lu_setup_per_col::Float64 = 400.0
end

"""
    lhl_step_ns(c::CostModel, b; solves_per_shift = 4, shifts_per_reduce = 25) -> ns

Modelled cost of one step — one shift, `solves_per_shift` solves, and `1/shifts_per_reduce`
of a reduction — of the dense LHL kernel on a block of size `b`.
"""
function lhl_step_ns(c::CostModel, b::Integer; solves_per_shift::Real = 4, shifts_per_reduce::Real = 25, factors = (1.0, 1.0, 1.0))
    bf = Float64(b)
    shift = factors[1] * (c.lhl_shift_per_b2 * bf^2 + c.lhl_shift_per_b * bf)
    solve = factors[2] * (c.lhl_solve_per_b2 * bf^2 + c.lhl_solve_per_b * bf)
    reduce = factors[3] * (c.lhl_reduce_per_b3 * bf^3 + c.lhl_reduce_per_b2 * bf^2)
    return shift + solves_per_shift * solve + reduce / max(shifts_per_reduce, 1)
end

"""
    lu_step_ns(c::CostModel, b, flops, nnz; solves_per_shift = 4, shifts_per_reduce = 25) -> ns

Modelled cost of one step — one refactorization, `solves_per_shift` solves, and
`1/shifts_per_reduce` of the first (pivoting) factorization — of the sparse LU kernel on a
block of size `b` whose refactorization costs `flops` multiply–adds and whose factors hold
`nnz` entries.
"""
function lu_step_ns(c::CostModel, b::Integer, flops::Real, nnz::Real; solves_per_shift::Real = 4, shifts_per_reduce::Real = 25, factors = (1.0, 1.0, 1.0))
    bf = Float64(b)
    refactor = factors[1] * (c.lu_refactor_per_flop * flops + c.lu_refactor_per_col * bf)
    solve = factors[2] * (c.lu_solve_per_nnz * nnz + c.lu_solve_per_row * bf)
    setup = factors[3] * (c.lu_setup_factor * refactor + c.lu_setup_per_col * bf)
    return refactor + solves_per_shift * solve + setup / max(shifts_per_reduce, 1)
end

"""
    BlockEstimate

Per-block numbers from the symbolic analysis: the block size; the modelled cost in
nanoseconds of one step (a shift, `solves_per_shift` solves and the amortized setup) with
the dense LHL kernel (`lhl_step`) and with the sparse LU kernel (`lu_step`); and the probed
LU's `nnz(L)`, `nnz(U)` and refactorization multiply–adds (`NaN` when the block was not
probed).
"""
struct BlockEstimate
    size::Int
    lhl_step::Float64
    lu_step::Float64
    lu_lnz::Float64
    lu_unz::Float64
    lu_flops::Float64
end

"""
    SparseLHLSymbolic

The shift- and value-independent part of a sparse LHL factorization: the symmetric block
triangular permutation, the kind of kernel every diagonal block uses, the gather maps from
the matrix's `nzval` into each block's storage, and the off-diagonal (strictly upper block
triangular) part in CSC over the permuted indices.  Built by [`slhl_analyze`](@ref).
"""
struct SparseLHLSymbolic
    n::Int
    nblocks::Int
    perm::Vector{Int}            # J[perm, perm] is block upper triangular
    iperm::Vector{Int}
    R::Vector{Int}               # block k is positions R[k]:R[k+1]-1
    kind::Vector{UInt8}
    kidx::Vector{Int}            # index of the block among those of its kind
    didx::Vector{Int}            # dense gather index (small and big blocks), 0 otherwise
    nscalar::Int
    nsmall::Int
    nbig::Int
    nlu::Int
    sdiag::Vector{Int}           # per scalar block: nz index of its diagonal (0 if absent)
    bmapptr::Vector{Int}         # per dense block: range in bmap/bpos
    bmap::Vector{Int}            # nz index in J
    bpos::Vector{Int}            # linear position in the b×b block
    voff::Vector{Int}            # per small block: offsets into vstore / gstore / istore / swap
    goff::Vector{Int}
    ioff::Vector{Int}
    soff::Vector{Int}
    vlen::Int
    glen::Int
    ilen::Int
    slen::Int
    maxbig::Int
    offp::Vector{Int}            # off-diagonal part, CSC over permuted columns
    offi::Vector{Int}            # permuted row indices
    offmap::Vector{Int}          # nz index in J
    lucolptr::Vector{Vector{Int}}  # per LU block: local CSC pattern, diagonal present
    lurowval::Vector{Vector{Int}}
    lumap::Vector{Vector{Int}}     # nz index in J, 0 for an inserted diagonal
    ludiag::Vector{Vector{Int}}    # per LU block: position of the diagonal in each column
    luscale::Int                   # PureKLU row scaling for the LU blocks (0 none, 1 sum, 2 max)
    lutol::Float64                 # PureKLU pivot tolerance for the LU blocks
    luprealloc::Int                # PureKLU fully_preallocated: -1 auto, 0 no, 1 yes
    lurepivot::Float64             # an LU block's pivot sequence is reused only while min|U|/max|U| exceeds this
    est::Vector{BlockEstimate}
end

"""
    slhl_analyze(J; kernel = :auto, small_max = 64, lu_min = 24, lhl_max = 4096,
                 solves_per_shift = 4, shifts_per_reduce = 25, costs = CostModel(),
                 probe = (1, 1), lu_scale = 0, lu_tol = 0.001,
                 lu_fully_preallocated = nothing, lu_repivot = nothing,
                 shift = eltype(J)) -> SparseLHLSymbolic

Symbolic analysis of a square sparse `J` for the shifted family `σI + τJ`: the symmetric
block triangular form (strongly connected components of `J`'s digraph, the BTF of any
`σI + τJ` with `σ ≠ 0`) and, per diagonal block, the kernel that will serve it.

  - `kernel = :lhl`: every block of size ≥ 2 is reduced to Hessenberg form (dense LHL);
    blocks up to `small_max` on the packed small-block storage, larger ones in an
    `LHLFactorization.LHLWorkspace`.
  - `kernel = :lu`: every block of size ≥ 2 is factored by a sparse LU (PureKLU) at every
    shift — the KLU strategy, inside the same framework.
  - `kernel = :auto`: blocks smaller than `lu_min` take the dense LHL kernel; a larger block
    is probed with one PureKLU factorization of `probe[1]·I + probe[2]·J` restricted to it,
    and the kernel with the smaller modelled cost per step wins, a step being one shift,
    `solves_per_shift` solves, and `1/shifts_per_reduce` of the one-time setup (the `O(b³)`
    Hessenberg reduction, or the LU's first factorization) — see [`CostModel`](@ref),
    [`lhl_step_ns`](@ref), [`lu_step_ns`](@ref).  A block larger than `lhl_max` is never
    given the dense kernel by `:auto` (its `O(b³)` reduction and `O(b²)` storage are what
    that would cost); `kernel = :lhl` forces it regardless.

`solves_per_shift` is how many right-hand sides a shift serves before the next shift, and
`shifts_per_reduce` how many shifts a reduction serves before `J` changes: an implicit ODE
solver does a handful of Newton solves per step and holds `J` for tens of steps.  The
analysis is pattern-based except for the probe, which uses `J`'s values for pivoting.  `J`
is not modified.

The LU blocks (and the probe) take PureKLU's options: `lu_scale` is its row scaling (`0`
none — the default here: it skips a pass in every refactorization and solve, measured
11–35 % faster refactorizations on low-fill blocks with unchanged backward errors, the
pivots then being chosen on the unscaled rows; `2` max-abs, KLU's own default, and `1` sum
are there for badly row-scaled blocks), `lu_tol` its pivot tolerance (`0.001` prefers the
diagonal, `1.0` is strict partial pivoting), and `lu_fully_preallocated` its upper-bound
preallocation (`nothing` = PureKLU's size heuristic; `true` guarantees that even a
re-pivoting factorization allocates nothing, at `O(b²)` memory per block — not for large
blocks).  With the heuristic (`nothing`), a block factored as sparse LU whose fill turns out
denser than the estimate grows its `L`/`U` once, inside the *first* shift — so the "first
shift allocates nothing" property is exact for genuinely sparse blocks and for `:auto` (which
sends near-dense blocks to the dense kernel), and holds from the second shift on otherwise;
`lu_fully_preallocated = true` makes even the first shift allocation-free.  The pivot sequence of an LU block is fixed at the first shift after each reduction
and reused by the following shifts as long as it came from a healthy factorization: a shift
that meets a zero pivot, or whose factors have `min|U_ii| / max|U_ii| < lu_repivot` (default
`eps^(3/4)`, about `10⁻¹²` for `Float64` — a pivot sequence chosen on a numerically singular
matrix is not trusted for the next one), re-pivots, and so does the shift after it.

`shift = Complex{real(eltype(J))}` tells the cost model that the shifts will be complex on a
real reduction (complex `O(b²)` shifts and solves, a complex LU, the same real reduction);
the analysis itself does not depend on it.

`J` must be a canonical `SparseMatrixCSC`: no duplicate row index within a column (what
`sparse` produces; a hand-built CSC with duplicates is rejected with an `ArgumentError`, since
the solve's scatter relies on distinct rows and PureKLU on a valid pattern).  Unsorted rows
within a column are fine.
"""
function slhl_analyze(
        J::SparseMatrixCSC{T}; kernel::Symbol = :auto, small_max::Int = 64, lu_min::Int = 24,
        lhl_max::Int = 4096, solves_per_shift::Real = 4.0, shifts_per_reduce::Real = 25.0,
        costs::CostModel = CostModel(), probe = (1, 1), lu_scale::Integer = 0, lu_tol::Real = 0.001,
        lu_fully_preallocated::Union{Bool, Nothing} = nothing, lu_repivot::Union{Real, Nothing} = nothing,
        shift::Type = T
    ) where {T}
    (shift === T || shift === Complex{real(T)}) ||
        throw(ArgumentError("shift must be $T or $(Complex{real(T)}), got $shift"))
    # a complex shift on a real reduction: complex O(b²) shifts and solves (≈ 3.5× / 2× the
    # real ones, LHLFactorization's measurements), a complex LU (≈ 4× the flops), the same real reduction
    cplx = shift <: Complex && !(T <: Complex)
    lhl_mul = cplx ? (3.5, 2.0, 1.0) : (1.0, 1.0, 1.0)
    lu_mul = cplx ? (4.0, 2.0, 4.0) : (1.0, 1.0, 1.0)
    kernel in (:auto, :lhl, :lu) ||
        throw(ArgumentError("kernel must be :auto, :lhl or :lu, got $kernel"))
    small_max >= 2 || throw(ArgumentError("small_max must be at least 2"))
    lu_scale in (0, 1, 2) || throw(ArgumentError("lu_scale must be 0, 1 or 2, got $lu_scale"))
    0 <= lu_tol <= 1 || throw(ArgumentError("lu_tol must lie in [0, 1], got $lu_tol"))
    luprealloc = lu_fully_preallocated === nothing ? -1 : Int(lu_fully_preallocated)
    lurepivot = lu_repivot === nothing ? Float64(eps(real(T))^(3 / 4)) : Float64(lu_repivot)
    0 <= lurepivot < 1 || throw(ArgumentError("lu_repivot must lie in [0, 1), got $lu_repivot"))
    n = checksquare(J)
    colptr = J.colptr
    rowval = J.rowval
    _check_pattern(n, colptr, rowval)
    perm, R = scc_order(n, colptr, rowval)
    nblocks = length(R) - 1
    iperm = invperm(perm)

    # Pass 1: in-block entries (local row, local column, nz index), contiguous per block and
    # sorted by column since columns are visited in permuted order; off-diagonal CSC.
    nnzJ = length(rowval)
    erow = Vector{Int}(undef, nnzJ)
    ecol = Vector{Int}(undef, nnzJ)
    enz = Vector{Int}(undef, nnzJ)
    eptr = Vector{Int}(undef, nblocks + 1)
    offp = Vector{Int}(undef, n + 1)
    offi = Int[]
    offmap = Int[]
    sizehint!(offi, nnzJ)
    sizehint!(offmap, nnzJ)
    ne = 0
    @inbounds for k in 1:nblocks
        eptr[k] = ne + 1
        k1 = R[k]
        k2 = R[k + 1]
        for c in k1:(k2 - 1)
            j = perm[c]
            offp[c] = length(offi) + 1
            for p in colptr[j]:(colptr[j + 1] - 1)
                ci = iperm[rowval[p]]
                if ci < k1
                    push!(offi, ci)
                    push!(offmap, p)
                elseif ci < k2
                    ne += 1
                    erow[ne] = ci - k1 + 1
                    ecol[ne] = c - k1 + 1
                    enz[ne] = p
                else
                    error("internal error: an entry below the block diagonal")
                end
            end
        end
    end
    eptr[nblocks + 1] = ne + 1
    offp[n + 1] = length(offi) + 1

    # Pass 2: kernel per block.
    kind = Vector{UInt8}(undef, nblocks)
    kidx = zeros(Int, nblocks)
    didx = zeros(Int, nblocks)
    est = Vector{BlockEstimate}(undef, nblocks)
    nscalar = nsmall = nbig = nlu = 0
    sdiag = Int[]
    nzval = J.nzval
    @inbounds for k in 1:nblocks
        b = R[k + 1] - R[k]
        if b == 1
            kind[k] = KIND_SCALAR
            nscalar += 1
            kidx[k] = nscalar
            d = 0
            for e in eptr[k]:(eptr[k + 1] - 1)
                d = enz[e]
            end
            push!(sdiag, d)
            est[k] = BlockEstimate(1, 0.0, 0.0, 0.0, 0.0, 0.0)
            continue
        end
        lhl_cost = lhl_step_ns(costs, b; solves_per_shift, shifts_per_reduce, factors = lhl_mul)
        uselu = false
        e = BlockEstimate(b, lhl_cost, NaN, NaN, NaN, NaN)
        if kernel == :lu || (kernel == :auto && b > lhl_max)
            uselu = true
        elseif kernel == :auto && b >= lu_min
            if b >= 128 && _narrow_band(b, erow, ecol, eptr[k], eptr[k + 1] - 1)
                # a narrow, nearly full band: its LU costs O(b·bw²) per shift while the
                # Hessenberg reduction fills it densely — the LU kernel wins outright from
                # b ≈ 100 on, so the probe factorization is skipped (PureKLU's band guards)
                uselu = true
            else
                lnz, unz, flops = _probe_block(T, b, erow, ecol, enz, eptr[k], eptr[k + 1] - 1, nzval, probe, Int(lu_scale), Float64(lu_tol))
                if !isnan(flops)
                    lu_cost = lu_step_ns(costs, b, flops, lnz + unz; solves_per_shift, shifts_per_reduce, factors = lu_mul)
                    e = BlockEstimate(b, lhl_cost, lu_cost, lnz, unz, flops)
                    uselu = lu_cost < lhl_cost
                end
            end
        end
        est[k] = e
        if uselu
            kind[k] = KIND_LU
            nlu += 1
            kidx[k] = nlu
        elseif b <= small_max
            kind[k] = KIND_SMALL
            nsmall += 1
            kidx[k] = nsmall
        else
            kind[k] = KIND_BIG
            nbig += 1
            kidx[k] = nbig
        end
    end

    # Pass 3: storage layout and gather maps.
    ndense = nsmall + nbig
    bmapptr = Vector{Int}(undef, ndense + 1)
    bmap = Int[]
    bpos = Int[]
    sizehint!(bmap, ne)
    sizehint!(bpos, ne)
    voff = Vector{Int}(undef, nsmall)
    goff = Vector{Int}(undef, nsmall)
    ioff = Vector{Int}(undef, nsmall)
    soff = Vector{Int}(undef, nsmall)
    lucolptr = Vector{Vector{Int}}(undef, nlu)
    lurowval = Vector{Vector{Int}}(undef, nlu)
    lumap = Vector{Vector{Int}}(undef, nlu)
    ludiag = Vector{Vector{Int}}(undef, nlu)
    vlen = glen = ilen = slen = 0
    maxbig = 0
    d = 0
    @inbounds for k in 1:nblocks
        b = R[k + 1] - R[k]
        kd = kind[k]
        if kd == KIND_SMALL || kd == KIND_BIG
            d += 1
            didx[k] = d
            bmapptr[d] = length(bmap) + 1
            for e in eptr[k]:(eptr[k + 1] - 1)
                push!(bmap, enz[e])
                push!(bpos, erow[e] + (ecol[e] - 1) * b)
            end
            if kd == KIND_SMALL
                s = kidx[k]
                voff[s] = vlen
                goff[s] = glen
                ioff[s] = ilen
                soff[s] = slen
                vlen += _sb_vlen(b)
                glen += _sb_glen(b)
                ilen += _sb_ilen(b)
                slen += _sb_slen(b)
            else
                maxbig = max(maxbig, b)
            end
        elseif kd == KIND_LU
            l = kidx[k]
            cp, rv, mp = _block_csc(b, erow, ecol, enz, eptr[k], eptr[k + 1] - 1)
            dg = Vector{Int}(undef, b)
            for j in 1:b, p in cp[j]:(cp[j + 1] - 1)
                rv[p] == j && (dg[j] = p)
            end
            lucolptr[l], lurowval[l], lumap[l], ludiag[l] = cp, rv, mp, dg
        end
    end
    bmapptr[ndense + 1] = length(bmap) + 1

    return SparseLHLSymbolic(
        n, nblocks, perm, iperm, R, kind, kidx, didx, nscalar, nsmall, nbig, nlu, sdiag,
        bmapptr, bmap, bpos, voff, goff, ioff, soff, vlen, glen, ilen, slen, maxbig, offp, offi, offmap,
        lucolptr, lurowval, lumap, ludiag, Int(lu_scale), Float64(lu_tol), luprealloc, lurepivot, est
    )
end

# Local CSC pattern of a block from its entry list (sorted by column), with rows sorted
# within each column and the diagonal inserted where it is structurally absent (map 0).
function _block_csc(b::Int, erow, ecol, enz, e1::Int, e2::Int)
    colptr = Vector{Int}(undef, b + 1)
    rowval = Int[]
    map = Int[]
    sizehint!(rowval, e2 - e1 + 1 + b)
    sizehint!(map, e2 - e1 + 1 + b)
    e = e1
    tmp = Tuple{Int, Int}[]
    @inbounds for j in 1:b
        colptr[j] = length(rowval) + 1
        empty!(tmp)
        hasdiag = false
        while e <= e2 && ecol[e] == j
            push!(tmp, (erow[e], enz[e]))
            erow[e] == j && (hasdiag = true)
            e += 1
        end
        hasdiag || push!(tmp, (j, 0))
        sort!(tmp; by = first)
        for (r, q) in tmp
            push!(rowval, r)
            push!(map, q)
        end
    end
    colptr[b + 1] = length(rowval) + 1
    return colptr, rowval, map
end

# One PureKLU factorization of probe[1]·I + probe[2]·J on the block: nnz(L), nnz(U) and the
# multiply–adds of a left-looking refactorization (Σᵢ |L(:,i)|·|U(i,:)| + nnz(L) + nnz(U)).
# NaN if the probe is numerically singular.
function _probe_block(::Type{T}, b::Int, erow, ecol, enz, e1::Int, e2::Int, nzval, probe, luscale::Int, lutol::Float64) where {T}
    colptr, rowval, map = _block_csc(b, erow, ecol, enz, e1, e2)
    σ0 = convert(T, probe[1])
    τ0 = convert(T, probe[2])
    vals = Vector{T}(undef, length(rowval))
    @inbounds for j in 1:b
        for p in colptr[j]:(colptr[j + 1] - 1)
            q = map[p]
            v = q == 0 ? zero(T) : τ0 * nzval[q]
            rowval[p] == j && (v += σ0)
            vals[p] = v
        end
    end
    S = SparseMatrixCSC{T, Int}(b, b, colptr, rowval, vals)
    K = PureKLU.klu(S; check = false, full_factor = false, tol = lutol)
    K.common.scale = Cint(luscale)       # the settings the block will be factored with
    PureKLU.klu_factor!(K; check = false, allowsingular = true)
    K.common.status == PureKLU.KLU_OK || return (NaN, NaN, NaN)
    return _klu_stats(K)
end

function _klu_stats(K::PureKLU.KLUFactorization)
    Num = getfield(K, :numeric)
    Sym = getfield(K, :symbolic)
    nb = Int(Sym.nblocks)
    R = Sym.R
    lnz = 0.0
    unz = 0.0
    flops = 0.0
    rc = zeros(Int, Int(Sym.maxblock))
    @inbounds for blk in 1:nb
        k1 = Int(R[blk])
        k2 = Int(R[blk + 1])
        nk = k2 - k1
        nk == 1 && continue
        LU = Num.LUbx[blk]
        fill!(view(rc, 1:nk), 0)
        for k in 0:(nk - 1)
            uip = Int(Num.Uip[k1 + k + 1])
            ulen = Int(Num.Ulen[k1 + k + 1])
            for p in 0:(ulen - 1)
                rc[Int(LU.Ui[uip + p + 1]) + 1] += 1
            end
            unz += ulen
            lnz += Int(Num.Llen[k1 + k + 1])
        end
        for i in 0:(nk - 1)
            flops += Int(Num.Llen[k1 + i + 1]) * rc[i + 1]
        end
    end
    flops += lnz + unz
    return lnz, unz, flops
end

Base.show(io::IO, s::SparseLHLSymbolic) = show(io, MIME"text/plain"(), s)
function Base.show(io::IO, ::MIME"text/plain", s::SparseLHLSymbolic)
    print(
        io, "SparseLHLSymbolic: $(s.n)×$(s.n), $(s.nblocks) blocks (",
        "$(s.nscalar) scalar, $(s.nsmall) small LHL, $(s.nbig) large LHL, $(s.nlu) sparse LU), ",
        "max block $(s.nblocks == 0 ? 0 : maximum(diff(s.R))), nnz(offdiag) = $(length(s.offi))"
    )
    return nothing
end

# A canonical CSC has distinct row indices within each column (`sparse` guarantees it; a
# hand-built one may not).  The solve's off-diagonal scatter and PureKLU's kernels rely on
# it, so it is checked once here, O(nnz), with a per-column marker.
function _check_pattern(n::Int, colptr::AbstractVector{<:Integer}, rowval::AbstractVector{<:Integer})
    length(colptr) == n + 1 || throw(ArgumentError("J's colptr has length $(length(colptr)), expected $(n + 1)"))
    mark = zeros(Int, n)
    @inbounds for j in 1:n
        for p in colptr[j]:(colptr[j + 1] - 1)
            i = Int(rowval[p])
            1 <= i <= n || throw(ArgumentError("J has a row index $i outside 1:$n"))
            mark[i] == j && throw(
                ArgumentError(
                    "J stores entry ($i, $j) more than once; combine the duplicates first " *
                        "(e.g. `sparse(findnz(J)..., size(J)...)`)"
                )
            )
            mark[i] = j
        end
    end
    return nothing
end

# A narrow, nearly full band (half-bandwidth ≤ 8, at least 4(bw+1) rows, at least half the
# band's slots stored) — PureKLU PR 48's three guards, which keep a sparse band such as a 2D
# Laplacian (bw ≈ √b, sparse inside the band) from being taken for one.
function _narrow_band(b::Int, erow, ecol, e1::Int, e2::Int)
    bw = 0
    @inbounds for e in e1:e2
        d = abs(erow[e] - ecol[e])
        d > 8 && return false
        bw = max(bw, d)
    end
    cnt = e2 - e1 + 1
    return b >= 4 * (bw + 1) && 2 * cnt >= b * (2 * bw + 1)
end
