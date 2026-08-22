# ---------------------------------------------------------------------------
# Symmetric block triangular form: strongly connected components
# ---------------------------------------------------------------------------

"""
    scc_order(n, colptr, rowval) -> (perm, R)

Strongly connected components of the digraph of a CSC pattern — an edge `j → i` for every
stored `(i, j)` — by Tarjan's algorithm with explicit stacks.  Components come out in
reverse topological order (the first one completed is a sink) and are numbered in that
order, so sorting the vertices by component number gives a symmetric permutation `perm`
under which `J[perm, perm]` is block **upper** triangular: every stored entry `(i, j)` has
`block(i) ≤ block(j)`.  Block `k` occupies positions `R[k]:R[k+1]-1`; `length(R) = nblocks + 1`.

This is the block triangular form of `σI + τJ`: the shift puts a structural nonzero on the
whole diagonal, so the maximum transversal is the identity and the BTF of KLU reduces to the
SCC decomposition of `J`'s digraph — and, unlike KLU's, the permutation is symmetric, which
a similarity transformation needs.  (A self-loop changes no component, so the diagonal need
not be present in the pattern.)
"""
function scc_order(n::Int, colptr::AbstractVector{<:Integer}, rowval::AbstractVector{<:Integer})
    index = zeros(Int, n)            # discovery time, 0 = unvisited
    low = Vector{Int}(undef, n)
    comp = zeros(Int, n)
    onstack = zeros(Bool, n)
    tstack = Vector{Int}(undef, n)   # Tarjan's stack
    dv = Vector{Int}(undef, n)       # DFS stack: vertex
    dp = Vector{Int}(undef, n)       # DFS stack: next edge position
    th = 0
    t = 0
    ncomp = 0
    @inbounds for root in 1:n
        index[root] == 0 || continue
        dh = 1
        dv[1] = root
        dp[1] = Int(colptr[root])
        t += 1
        index[root] = t
        low[root] = t
        th += 1
        tstack[th] = root
        onstack[root] = true
        while dh > 0
            v = dv[dh]
            p = dp[dh]
            pend = Int(colptr[v + 1])
            descended = false
            while p < pend
                w = Int(rowval[p])
                p += 1
                if index[w] == 0
                    dp[dh] = p
                    dh += 1
                    dv[dh] = w
                    dp[dh] = Int(colptr[w])
                    t += 1
                    index[w] = t
                    low[w] = t
                    th += 1
                    tstack[th] = w
                    onstack[w] = true
                    descended = true
                    break
                elseif onstack[w]
                    lw = index[w]
                    lw < low[v] && (low[v] = lw)
                end
            end
            descended && continue
            if low[v] == index[v]
                ncomp += 1
                while true
                    w = tstack[th]
                    th -= 1
                    onstack[w] = false
                    comp[w] = ncomp
                    w == v && break
                end
            end
            dh -= 1
            if dh > 0
                u = dv[dh]
                lv = low[v]
                lv < low[u] && (low[u] = lv)
            end
        end
    end
    # counting sort of the vertices by component
    R = zeros(Int, ncomp + 1)
    @inbounds for v in 1:n
        R[comp[v] + 1] += 1
    end
    R[1] = 1
    @inbounds for k in 1:ncomp
        R[k + 1] += R[k]
    end
    perm = Vector{Int}(undef, n)
    pos = copy(R)
    @inbounds for v in 1:n
        c = comp[v]
        perm[pos[c]] = v
        pos[c] += 1
    end
    return perm, R
end
