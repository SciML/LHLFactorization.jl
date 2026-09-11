using LHLFactorization, BenchmarkTools
using StableRNGs, LinearAlgebra

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# LHL factorizes the bordered system W = [J L; L' H]
function make_system(n)
    J = rand(rng, n, n) + n * I
    L = rand(rng, n, 4)
    H = Matrix(Symmetric(rand(rng, 4, 4)))
    W = [J L; L' H]
    return W, J
end

W100, J100 = make_system(100)
W500, J500 = make_system(500)
b100 = rand(rng, 104)
b500 = rand(rng, 504)

# =============================================================================
# Factorization
# =============================================================================

SUITE["factorize"] = BenchmarkGroup()

SUITE["factorize"]["lhl_100"] = @benchmarkable lhl($W100)
SUITE["factorize"]["lhl_500"] = @benchmarkable lhl($W500)

ws100 = lhl(W100)
ws500 = lhl(W500)

# =============================================================================
# Solve via ldiv
# =============================================================================

SUITE["ldiv"] = BenchmarkGroup()

SUITE["ldiv"]["ldiv_100"] = @benchmarkable lhl_ldiv!(copy($b100), $ws100)
SUITE["ldiv"]["ldiv_500"] = @benchmarkable lhl_ldiv!(copy($b500), $ws500)

# =============================================================================
# Shift / refactor
# =============================================================================

SUITE["shift"] = BenchmarkGroup()

SUITE["shift"]["lhl_shift_100"] = @benchmarkable lhl_shift!(ws, 0.5, 0.1) setup = (
    ws = lhl($W100)
)
SUITE["shift"]["lhl_reduce_100"] = @benchmarkable lhl_reduce!(ws, $W100, false) setup = (
    ws = lhl($W100)
)
