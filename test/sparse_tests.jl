# Sparse LHL — the `LHLFactorizationSparseExt` extension (SparseArrays + PureKLU).  Solves
# the shifted family `(σI + τJ)x = b` for a sparse `J` through the package verbs `lhl`,
# `lhl_shift!`, `lhl_ldiv!`, `lhl_refine!`, `lhl!`.
using LHLFactorization, SparseArrays, PureKLU, LinearAlgebra, Random, Test

@test Base.get_extension(LHLFactorization, :LHLFactorizationSparseExt) !== nothing

# a block upper triangular matrix with irreducible diagonal blocks of the given sizes,
# symmetrically permuted at random (so the solver must recover the block structure)
function btf_matrix(sizes; cpl = 0.05, density = 0.5, T = Float64, rng = Random.default_rng())
    n = sum(sizes)
    I_ = Int[]
    J_ = Int[]
    V = T[]
    off = 0
    for b in sizes
        if b == 1
            push!(I_, off + 1)
            push!(J_, off + 1)
            push!(V, randn(rng, T))
        else
            for j in 1:b, i in 1:b
                (i == j || rand(rng) < density) || continue
                push!(I_, off + i)
                push!(J_, off + j)
                push!(V, randn(rng, T))
            end
            for i in 1:(b - 1)
                push!(I_, off + i + 1)
                push!(J_, off + i)
                push!(V, randn(rng, T))
            end
            push!(I_, off + 1)
            push!(J_, off + b)
            push!(V, randn(rng, T))
        end
        if off > 0
            for _ in 1:max(1, round(Int, cpl * off * b))
                push!(I_, rand(rng, 1:off))
                push!(J_, off + rand(rng, 1:b))
                push!(V, randn(rng, T))
            end
        end
        off += b
    end
    J = sparse(I_, J_, V, n, n)
    p = randperm(rng, n)
    return J[p, p]
end

bwd(A, x, b) = norm(A * x - b) / (opnorm(A, 1) * norm(x) + norm(b))

function check(J, F; shifts = ((1.0, -0.1), (1.0, -0.02), (0.3, 1.0), (2.0, 0.5)), tol = 1.0e-10)
    n = size(J, 1)
    TT = eltype(J)
    for (σ, τ) in shifts
        lhl_shift!(F, σ, τ)
        @test F.info == 0
        b = randn(TT, n)
        x = F \ b
        A = σ * I + τ * J
        @test bwd(A, x, b) < tol
        y = similar(b)
        ldiv!(y, F, b)
        @test y == x
        x2 = copy(x)
        lhl_refine!(x2, A, b, F, 1)
        @test bwd(A, x2, b) < tol
    end
    return
end

Random.seed!(20260822)

@testset "every kernel, mixed block structure, dispatched through `lhl`" begin
    J = btf_matrix([1, 1, 1, 3, 5, 2, 8, 30, 1, 60, 4, 1, 100, 7, 20])
    for kernel in (:lhl, :lu, :auto), small_max in (48, 4, 1000)
        F = lhl(J; kernel, small_max, lu_min = 20)
        @test F isa LinearAlgebra.Factorization
        check(J, F)
        @test sprint(show, MIME"text/plain"(), F) isa String
    end
end

@testset "one irreducible block (2D grid), small and large paths" begin
    N = 12
    L1 = spdiagm(0 => fill(-2.0, N), 1 => ones(N - 1), -1 => ones(N - 1))
    J = kron(I(N), L1) + kron(L1, I(N)) + 0.3 * sprandn(N^2, N^2, 0.01)
    for kernel in (:lhl, :lu, :auto), small_max in (48, 200)
        F = lhl(J; kernel, small_max)
        check(J, F)
    end
end

@testset "matrix right-hand sides and a complex right-hand side" begin
    J = btf_matrix([1, 3, 5, 30, 60, 4, 20])
    n = size(J, 1)
    F = lhl(J; kernel = :auto, lu_min = 20)
    lhl_shift!(F, 1.0, -0.1)
    A = I - 0.1J
    B = randn(n, 5)
    X = F \ B
    @test X isa Matrix{Float64}
    for j in 1:5
        @test bwd(A, X[:, j], B[:, j]) < 1.0e-10
        @test X[:, j] ≈ (F \ B[:, j])
    end
    bc = randn(ComplexF64, n)      # complex RHS on a real factorization, by channels
    xc = F \ bc
    @test bwd(A, xc, bc) < 1.0e-10
end

@testset "complex shifts on a real reduction" begin
    J = btf_matrix([1, 3, 5, 30, 60, 20])
    n = size(J, 1)
    for kernel in (:lhl, :lu, :auto)
        F = lhl(J; kernel, shift = ComplexF64, lu_min = 20)
        for (σ, τ) in ((1.0 + 0.5im, -0.1), (0.5im, 1.0))
            lhl_shift!(F, σ, τ)
            b = randn(ComplexF64, n)
            x = F \ b
            @test bwd(σ * I + τ * J, x, b) < 1.0e-10
        end
    end
end

@testset "re-reduce with new values, and singular shift reported not thrown" begin
    J = btf_matrix([3, 5, 1, 20, 60, 1, 1, 12])
    F = lhl(J; kernel = :auto)
    J2 = copy(J)
    J2.nzval .= randn(nnz(J2))
    lhl!(F, J2)
    check(J2, F)
    # a structurally singular matrix: info reports it, no throw
    Jz = copy(J)
    Jz[:, 1] .= 0
    dropzeros!(Jz)
    Fz = lhl(Jz; kernel = :lhl)
    lhl_shift!(Fz, 0.0, 1.0)
    @test Fz.info != 0
    @test !LinearAlgebra.issuccess(Fz)
end

@testset "threaded shift/reduce is bit-identical to serial" begin
    J = btf_matrix([1, 40, 3, 80, 1, 60, 100])
    n = size(J, 1)
    for kernel in (:lhl, :lu, :auto)
        Fs = lhl(J; kernel, lu_min = 20, thread = false)
        Ft = lhl(J; kernel, lu_min = 20, thread = true)
        @test Ft.vstore == Fs.vstore
        for (σ, τ) in ((1.0, -0.1), (0.3, 1.0))
            lhl_shift!(Fs, σ, τ)
            lhl_shift!(Ft, σ, τ)
            @test Ft.gstore == Fs.gstore
            @test Ft.info == Fs.info
            b = randn(n)
            @test (Ft \ b) == (Fs \ b)
        end
    end
end

@testset "consumer hooks: lhl_isreduced and lhl_prefers_sparse" begin
    J = btf_matrix([1, 3, 5, 30, 4])       # reducible: several blocks
    F = lhl(J)
    @test lhl_isreduced(F)
    @test lhl_prefers_sparse(J)
    # one big irreducible block: KLU's regime, not preferred
    Jbig = btf_matrix([200])
    @test !lhl_prefers_sparse(Jbig)
    @test lhl_prefers_sparse(Jbig; lhl_max = 500) == false   # still one block
    @test lhl_prefers_sparse(sparse(2.0I, 5, 5))              # diagonal: five 1×1 blocks, reducible
end

@testset "Int32 indices and Float32" begin
    J = btf_matrix([1, 3, 5, 20, 1, 40])
    F = lhl(SparseMatrixCSC{Float64, Int32}(J); kernel = :auto, lu_min = 10)
    check(J, F)
    Jf = btf_matrix([3, 5, 1, 20, 40]; T = Float32)
    check(Jf, lhl(Jf; kernel = :lhl); tol = 1.0e-4)
end
