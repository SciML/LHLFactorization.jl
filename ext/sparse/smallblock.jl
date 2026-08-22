# ---------------------------------------------------------------------------
# Dense LHL kernels for small blocks, on flat storage.
#
# Every small block of size b owns, inside one `Vector{T}` (`vstore`), a contiguous region
# laid out as
#
#     factors  b×b   column-major; after the reduction, the step-k multipliers in
#                    rows k+2:b of column k (the Hessenberg part is not used from here)
#     Ht       b×b   H transposed: Ht[j, i] = H[i, j] for j ≥ i-1 (the rest is never read)
#     scale    b     balancing D (powers of two) and
#     iscale   b     its reciprocals
#
# and, in a second vector `gstore` of the *shift's* element type `TG` (the reduction's `T`,
# or `Complex{T}` for complex shifts on a real reduction),
#
#     Gt       b×b   the LU of the shifted Hessenberg σI + τH, transposed, with the
#                    multiplier of step k in the structural zero Gt[k, k+1]
#     rdiag    b     reciprocals of the pivots of Gt
#
# and in `istore`: ipiv (b-2), perm (b), iperm (b); in `swap`: b flags.  The transposed
# layouts make every inner loop of the shift and of the back substitution contiguous,
# exactly as in LHLFactorization's `Gt`/`Ht`.  The shift's pending row and the solve's
# vector live in a separate scratch vector `y` (shared by all small blocks), so that no hot
# loop reads and writes the same array.  All index arithmetic is explicit so that the
# kernels run on the flat vectors without views.  The L sweeps take four steps per pass and
# the back substitution four rows per pass, as LHLFactorization's do; the solve kernels
# spell their multiply–adds as `muladd` (Julia does not contract `a*b + c` on its own, and
# the serial recurrences are the critical path — measured −11 % on the solve at b = 8..32;
# the shift's loops are left alone, where it measured slower).
#
# `_sb_solve!` and `_sb_shift!` take the block size either as an `Int` or as a `Val{B}`:
# with a constant `B` the cores below unroll completely, which is what the drivers use for
# b ≤ 8 (solve) and b ≤ 4 (shift), where loop overheads dominate.  Same algorithm and
# operation count; the `@simd` reductions of the back substitution may associate
# differently for a constant trip count, so the two forms agree to rounding, not bitwise
# (the shift forms are bitwise identical).
# ---------------------------------------------------------------------------

@inline _sb_vlen(b::Int) = 2 * b * b + 2 * b
@inline _sb_glen(b::Int) = b * b + b
@inline _sb_ilen(b::Int) = 3 * b - 2
@inline _sb_slen(b::Int) = b

# offsets of the regions inside the block's value region starting at `o` (vstore) and its
# shift region starting at `g` (gstore)
@inline _sb_oHt(o::Int, b::Int) = o + b * b
@inline _sb_osc(o::Int, b::Int) = o + 2 * b * b
@inline _sb_oisc(o::Int, b::Int) = o + 2 * b * b + b
@inline _sb_gGt(g::Int, b::Int) = g
@inline _sb_grd(g::Int, b::Int) = g + b * b
@inline _sb_oip(io::Int, b::Int) = io
@inline _sb_operm(io::Int, b::Int) = io + b - 2
@inline _sb_oiperm(io::Int, b::Int) = io + 2b - 2

# Parlett–Reinsch balancing of the b×b block at `o` by powers of two (exact).
function _sb_balance!(vs::Vector{T}, o::Int, b::Int) where {T}
    Tr = real(T)
    osc = _sb_osc(o, b)
    oisc = _sb_oisc(o, b)
    @inbounds for i in 1:b
        vs[osc + i] = one(T)
        vs[oisc + i] = one(T)
    end
    @inbounds for _ in 1:20
        converged = true
        for i in 1:b
            # full row and column 1-norms with the diagonal taken out afterwards (branch-free;
            # the subtraction cancels when the diagonal dominates by ~1/eps, which then skips a
            # row the exclusion form would have scaled — measured accuracy-neutral or better)
            c = zero(Tr)
            r = zero(Tr)
            @simd for j in 1:b
                c += abs(vs[o + j + (i - 1) * b])
                r += abs(vs[o + i + (j - 1) * b])
            end
            d = abs(vs[o + i + (i - 1) * b])
            c -= d
            r -= d
            (c <= zero(Tr) || r <= zero(Tr)) && continue
            f = one(Tr)
            s = c + r
            while c < r / 2
                c *= 2
                r /= 2
                f *= 2
            end
            while c >= 2r
                c /= 2
                r *= 2
                f /= 2
            end
            if c + r < Tr(0.95) * s
                converged = false
                vs[osc + i] *= f
                vs[oisc + i] /= f
                for j in 1:b
                    vs[o + i + (j - 1) * b] /= f
                    vs[o + j + (i - 1) * b] *= f
                end
            end
        end
        converged && break
    end
    return nothing
end

# Wilkinson's elimination (ELMHES) with partial pivoting on the b×b block at `o`, in place:
# H ends up in the upper Hessenberg part, the multipliers below it; pivots in `ipiv`, the
# composed permutation in `perm`/`iperm`, and H transposed into the `Ht` region.  The
# rank-1 updates read column k and write a different column, so their ranges never
# overlap and `ivdep` spares LLVM's runtime overlap check.
function _sb_reduce!(vs::Vector{T}, o::Int, b::Int, is::Vector{Int}, io::Int) where {T}
    oip = _sb_oip(io, b)
    operm = _sb_operm(io, b)
    oiperm = _sb_oiperm(io, b)
    oHt = _sb_oHt(o, b)
    @inbounds begin
        for k in 1:(b - 2)
            ck = o + (k - 1) * b          # column k: A[i, k] = vs[ck + i]
            p = k + 1
            amax = abs(vs[ck + k + 1])
            for i in (k + 2):b
                a = abs(vs[ck + i])
                if a > amax
                    amax = a
                    p = i
                end
            end
            is[oip + k] = p
            if p != k + 1
                for j in 1:b                # rows k+1 ↔ p
                    i1 = o + (k + 1) + (j - 1) * b
                    i2 = o + p + (j - 1) * b
                    vs[i1], vs[i2] = vs[i2], vs[i1]
                end
                c1 = o + k * b              # columns k+1 ↔ p
                c2 = o + (p - 1) * b
                for i in 1:b
                    vs[c1 + i], vs[c2 + i] = vs[c2 + i], vs[c1 + i]
                end
            end
            piv = vs[ck + k + 1]
            iszero(piv) && continue
            for i in (k + 2):b
                vs[ck + i] /= piv
            end
            ck1 = o + k * b                 # column k+1
            pk = vs[ck1 + k + 1]
            if !iszero(pk)
                @simd ivdep for i in (k + 2):b
                    vs[ck1 + i] -= vs[ck + i] * pk
                end
            end
            for j in (k + 2):b
                cj = o + (j - 1) * b
                pj = vs[cj + k + 1]
                if !iszero(pj)
                    @simd ivdep for i in (k + 2):b
                        vs[cj + i] -= vs[ck + i] * pj
                    end
                end
                vj = vs[ck + j]
                if !iszero(vj)
                    @simd ivdep for i in 1:b
                        vs[ck1 + i] += vj * vs[cj + i]
                    end
                end
            end
        end
        for i in 1:b
            is[operm + i] = i
        end
        for k in 1:(b - 2)
            p = is[oip + k]
            is[operm + k + 1], is[operm + p] = is[operm + p], is[operm + k + 1]
        end
        for i in 1:b
            is[oiperm + is[operm + i]] = i
        end
        for i in 1:b
            for j in max(i - 1, 1):b
                vs[oHt + j + (i - 1) * b] = vs[o + i + (j - 1) * b]
            end
        end
    end
    return nothing
end

# LU of G = σI + τH with row-pair pivoting, into Gt; returns the index of the first zero
# pivot (0 if none).  Row k+1 of G is formed from Ht only when it enters the elimination and
# the not-yet-chosen row lives in `r` (scratch, length ≥ b), as in LHLFactorization's
# `_lhl_shift_rows!`.
@inline function _sb_shift_core!(
        vs::Vector{T}, o::Int, gs::Vector{TG}, g::Int, b::Int, sw::Vector{Bool}, so::Int,
        r::Vector{TG}, σ::TG, τ::TG
    ) where {T, TG}
    oHt = _sb_oHt(o, b)
    oGt = _sb_gGt(g, b)
    ord = _sb_grd(g, b)
    info = 0
    @inbounds begin
        @simd ivdep for j in 1:b
            r[j] = τ * vs[oHt + j]
        end
        r[1] += σ
        for k in 1:(b - 1)
            a = r[k]
            hk1 = oHt + k * b               # column k+1 of Ht = row k+1 of H
            bb = τ * vs[hk1 + k]            # G[k+1, k]
            gk = oGt + (k - 1) * b          # column k of Gt = row k of U
            if abs(bb) > abs(a)
                sw[so + k] = true
                gs[gk + k] = bb
                l = a / bb
                gs[oGt + k + k * b] = l
                @simd ivdep for j in (k + 1):b
                    gj = τ * vs[hk1 + j]
                    gs[gk + j] = gj
                    r[j] -= l * gj
                end
                gs[gk + k + 1] += σ
                r[k + 1] -= l * σ
            else
                sw[so + k] = false
                gs[gk + k] = a
                if iszero(a)
                    info == 0 && (info = k)
                    l = zero(TG)
                else
                    l = bb / a
                end
                gs[oGt + k + k * b] = l
                @simd ivdep for j in (k + 1):b
                    rj = r[j]
                    gs[gk + j] = rj
                    r[j] = τ * vs[hk1 + j] - l * rj
                end
                r[k + 1] += σ
            end
        end
        rn = r[b]
        gs[oGt + b + (b - 1) * b] = rn
        sw[so + b] = false
        iszero(rn) && info == 0 && (info = b)
        for j in 1:b
            gs[ord + j] = inv(gs[oGt + j + (j - 1) * b])
        end
    end
    return info
end
# (inlined into the driver as well: measured ~4 % faster than a call for b = 8..16)
@inline _sb_shift!(vs::Vector{T}, o::Int, gs::Vector{TG}, g::Int, b::Int, sw::Vector{Bool}, so::Int, r::Vector{TG}, σ::TG, τ::TG) where {T, TG} =
    _sb_shift_core!(vs, o, gs, g, b, sw, so, r, σ, τ)
@inline _sb_shift!(vs::Vector{T}, o::Int, gs::Vector{TG}, g::Int, ::Val{B}, sw::Vector{Bool}, so::Int, r::Vector{TG}, σ::TG, τ::TG) where {T, TG, B} =
    _sb_shift_core!(vs, o, gs, g, B, sw, so, r, σ, τ)

# y ← L⁻¹ y on the multipliers stored in the b×b block at `o`: four steps per pass, the
# head of a group a three-step recurrence, then rows k+5:b take all four columns at once.
@inline function _sb_linv!(y::Vector{TG}, vs::Vector{T}, o::Int, b::Int) where {T, TG}
    G = max(b - 2, 0) >> 2
    @inbounds begin
        for g in 1:G
            k = 4g - 3
            c1 = o + (k - 1) * b
            c2 = c1 + b
            c3 = c2 + b
            c4 = c3 + b
            x1 = y[k + 1]
            x2 = muladd(-vs[c1 + k + 2], x1, y[k + 2])
            y[k + 2] = x2
            x3 = muladd(-vs[c2 + k + 3], x2, muladd(-vs[c1 + k + 3], x1, y[k + 3]))
            y[k + 3] = x3
            x4 = muladd(-vs[c3 + k + 4], x3, muladd(-vs[c2 + k + 4], x2, muladd(-vs[c1 + k + 4], x1, y[k + 4])))
            y[k + 4] = x4
            @simd ivdep for i in (k + 5):b
                y[i] = muladd(-vs[c4 + i], x4, muladd(-vs[c3 + i], x3, muladd(-vs[c2 + i], x2, muladd(-vs[c1 + i], x1, y[i]))))
            end
        end
        for k in (4G + 1):(b - 2)
            ck = o + (k - 1) * b
            xk = y[k + 1]
            @simd ivdep for i in (k + 2):b
                y[i] = muladd(-vs[ck + i], xk, y[i])
            end
        end
    end
    return nothing
end

# y ← L y: the steps in reverse order; a group reads its four scalars first, since step
# k+c only touches rows k+c+2:b.
@inline function _sb_l!(y::Vector{TG}, vs::Vector{T}, o::Int, b::Int) where {T, TG}
    G = max(b - 2, 0) >> 2
    @inbounds begin
        for k in (b - 2):-1:(4G + 1)
            ck = o + (k - 1) * b
            xk = y[k + 1]
            @simd ivdep for i in (k + 2):b
                y[i] = muladd(vs[ck + i], xk, y[i])
            end
        end
        for g in G:-1:1
            k = 4g - 3
            c1 = o + (k - 1) * b
            c2 = c1 + b
            c3 = c2 + b
            c4 = c3 + b
            x1 = y[k + 1]
            x2 = y[k + 2]
            x3 = y[k + 3]
            x4 = y[k + 4]
            y[k + 2] = muladd(vs[c1 + k + 2], x1, x2)
            y[k + 3] = muladd(vs[c2 + k + 3], x2, muladd(vs[c1 + k + 3], x1, x3))
            y[k + 4] = muladd(vs[c3 + k + 4], x3, muladd(vs[c2 + k + 4], x2, muladd(vs[c1 + k + 4], x1, x4)))
            @simd ivdep for i in (k + 5):b
                y[i] = muladd(vs[c4 + i], x4, muladd(vs[c3 + i], x3, muladd(vs[c2 + i], x2, muladd(vs[c1 + i], x1, y[i]))))
            end
        end
    end
    return nothing
end

# The Hessenberg solve on the block's Gt: the forward sweep as selects, then back
# substitution four rows at a time — the dot products of the next block over y[j+1:b] do
# not depend on the current block's four unknowns, so they are issued right after its 4×4
# triangle (LHLFactorization's `_hessenberg_solve!`).
@inline function _sb_hess!(y::Vector{TG}, gs::Vector{TG}, g::Int, b::Int, sw::Vector{Bool}, so::Int) where {TG}
    T = TG
    oGt = _sb_gGt(g, b)
    ord = _sb_grd(g, b)
    @inbounds begin
        for k in 1:(b - 1)
            s = sw[so + k]
            a = y[k]
            c = y[k + 1]
            xk = ifelse(s, c, a)
            y[k] = xk
            y[k + 1] = muladd(-gs[oGt + k + k * b], xk, ifelse(s, a, c))
        end
        j = b
        s1 = zero(T)
        s2 = zero(T)
        s3 = zero(T)
        s4 = zero(T)
        while j - 3 >= 1
            g0 = oGt + (j - 1) * b        # column j of Gt
            g1 = g0 - b                   # column j-1
            g2 = g1 - b
            g3 = g2 - b
            xj = (y[j] - s1) * gs[ord + j]
            y[j] = xj
            s2 = muladd(gs[g1 + j], xj, s2)
            s3 = muladd(gs[g2 + j], xj, s3)
            s4 = muladd(gs[g3 + j], xj, s4)
            xj1 = (y[j - 1] - s2) * gs[ord + j - 1]
            y[j - 1] = xj1
            s3 = muladd(gs[g2 + j - 1], xj1, s3)
            s4 = muladd(gs[g3 + j - 1], xj1, s4)
            xj2 = (y[j - 2] - s3) * gs[ord + j - 2]
            y[j - 2] = xj2
            s4 = muladd(gs[g3 + j - 2], xj2, s4)
            xj3 = (y[j - 3] - s4) * gs[ord + j - 3]
            y[j - 3] = xj3
            jn = j - 4
            if jn - 3 >= 1
                h0 = oGt + (jn - 1) * b
                h1 = h0 - b
                h2 = h1 - b
                h3 = h2 - b
                t1 = zero(T)
                t2 = zero(T)
                t3 = zero(T)
                t4 = zero(T)
                @simd for i in (j + 1):b
                    xi = y[i]
                    t1 = muladd(gs[h0 + i], xi, t1)
                    t2 = muladd(gs[h1 + i], xi, t2)
                    t3 = muladd(gs[h2 + i], xi, t3)
                    t4 = muladd(gs[h3 + i], xi, t4)
                end
                s1 = muladd(gs[h0 + j - 3], xj3, muladd(gs[h0 + j - 2], xj2, muladd(gs[h0 + j - 1], xj1, muladd(gs[h0 + j], xj, t1))))
                s2 = muladd(gs[h1 + j - 3], xj3, muladd(gs[h1 + j - 2], xj2, muladd(gs[h1 + j - 1], xj1, muladd(gs[h1 + j], xj, t2))))
                s3 = muladd(gs[h2 + j - 3], xj3, muladd(gs[h2 + j - 2], xj2, muladd(gs[h2 + j - 1], xj1, muladd(gs[h2 + j], xj, t3))))
                s4 = muladd(gs[h3 + j - 3], xj3, muladd(gs[h3 + j - 2], xj2, muladd(gs[h3 + j - 1], xj1, muladd(gs[h3 + j], xj, t4))))
            end
            j = jn
        end
        while j >= 1
            gj = oGt + (j - 1) * b
            s = zero(T)
            @simd for i in (j + 1):b
                s = muladd(gs[gj + i], y[i], s)
            end
            y[j] = (y[j] - s) * gs[ord + j]
            j -= 1
        end
    end
    return nothing
end

# x ← (σI + τH_block)⁻¹ x on X[x0+1:x0+b]: Z⁻¹ (scale, permute, L⁻¹), the Hessenberg
# solve, Z (L, permute, scale), through the scratch `y`.
@inline function _sb_solve_core!(
        vs::Vector{T}, o::Int, gs::Vector{TG}, g::Int, b::Int, is::Vector{Int}, io::Int,
        sw::Vector{Bool}, so::Int, y::Vector{TG}, X::AbstractVector, x0::Int
    ) where {T, TG}
    osc = _sb_osc(o, b)
    oisc = _sb_oisc(o, b)
    operm = _sb_operm(io, b)
    oiperm = _sb_oiperm(io, b)
    @inbounds begin
        for i in 1:b
            p = is[operm + i]
            y[i] = X[x0 + p] * vs[oisc + p]
        end
        _sb_linv!(y, vs, o, b)
        _sb_hess!(y, gs, g, b, sw, so)
        _sb_l!(y, vs, o, b)
        for i in 1:b
            X[x0 + i] = y[is[oiperm + i]] * vs[osc + i]
        end
    end
    return nothing
end
function _sb_solve!(
        vs::Vector{T}, o::Int, gs::Vector{TG}, g::Int, b::Int, is::Vector{Int}, io::Int,
        sw::Vector{Bool}, so::Int, y::Vector{TG}, X::AbstractVector, x0::Int
    ) where {T, TG}
    return _sb_solve_core!(vs, o, gs, g, b, is, io, sw, so, y, X, x0)
end
@inline function _sb_solve!(
        vs::Vector{T}, o::Int, gs::Vector{TG}, g::Int, ::Val{B}, is::Vector{Int}, io::Int,
        sw::Vector{Bool}, so::Int, y::Vector{TG}, X::AbstractVector, x0::Int
    ) where {T, TG, B}
    return _sb_solve_core!(vs, o, gs, g, B, is, io, sw, so, y, X, x0)
end

# The drivers dispatch on the block size: blocks of 2..8 solve and 2..4 shift through the
# fully unrolled `Val` cores (measured: solve −20..−40 % per block, shift −10..−55 %),
# larger ones through the generic loops.
@inline function _sb_solve_dispatch!(vs::Vector{T}, o::Int, gs::Vector{TG}, g::Int, b::Int, args...) where {T, TG}
    b == 2 && return _sb_solve!(vs, o, gs, g, Val(2), args...)
    b == 3 && return _sb_solve!(vs, o, gs, g, Val(3), args...)
    b == 4 && return _sb_solve!(vs, o, gs, g, Val(4), args...)
    b == 5 && return _sb_solve!(vs, o, gs, g, Val(5), args...)
    b == 6 && return _sb_solve!(vs, o, gs, g, Val(6), args...)
    b == 7 && return _sb_solve!(vs, o, gs, g, Val(7), args...)
    b == 8 && return _sb_solve!(vs, o, gs, g, Val(8), args...)
    return _sb_solve!(vs, o, gs, g, b, args...)
end
@inline function _sb_shift_dispatch!(vs::Vector{T}, o::Int, gs::Vector{TG}, g::Int, b::Int, args...) where {T, TG}
    b == 2 && return _sb_shift!(vs, o, gs, g, Val(2), args...)
    b == 3 && return _sb_shift!(vs, o, gs, g, Val(3), args...)
    b == 4 && return _sb_shift!(vs, o, gs, g, Val(4), args...)
    return _sb_shift!(vs, o, gs, g, b, args...)
end
