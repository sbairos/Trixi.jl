using Trixi
using OrdinaryDiffEqSDIRK

# Functionality for automatic sparsity detection
using SparseDiffTools, Symbolics

# We need to avoid if-clauses to be able to use `Num` type from Symbolics.
# In the Trixi implementation, we overload the sqrt function to first check if the argument 
# is < 0 and then return NaN instead of an error.
# To turn off this behaviour, we switch back to the Base implementation here.
Trixi.set_sqrt_type!("sqrt_Base")

import Base: eps, zero, one, * # For overloading with type `Real`

###############################################################################
### Hacks ###

# Required for setting up the Lobatto-Legendre basis for abstract `Real` type
eps(::Type{Real}, RealT = Float64) = eps(RealT)

# There are some places where `one(RealT)` or `zero(uEltype)` is called where `RealT` or `uEltype` is `Real`.
# This returns an `Int64`, i.e., `1` or `0`, respectively.
# We don't want `Int`s for the sparsity detection, so we override this behavior.
one(::Type{Real}, RealT = Float64) = Base.one(RealT)
zero(::Type{Real}, RealT = Float64) = Base.zero(RealT)

# Multiplying two Matrix{Real}s gives a Matrix{Any}.
# This causes problems when instantiating the Legendre basis, which calls
# `calc_{forward,reverse}_{upper, lower}` which in turn uses the matrix multiplication
# which is overloaded here in construction of the interpolation/projection operators 
# required for mortars.
function *(A::Matrix{Real}, B::Matrix{Real})::Matrix{Real}
    m, n = size(A, 1), size(B, 2)
    kA = size(A, 2)
    kB = size(B, 1)
    @assert kA==kB "Matrix dimensions must match for multiplication"

    C = Matrix{Real}(undef, m, n)
    for i in 1:m, j in 1:n
        #acc::Real = zero(promote_type(typeof(A[i,1]), typeof(B[1,j])))
        acc = zero(Real)
        for k in 1:kA
            acc += A[i, k] * B[k, j]
        end
        C[i, j] = acc
    end
    return C
end

###############################################################################
### equations and solver ###

equations = CompressibleEulerEquations2D(1.4)

# For sparsity detection we can only use `flux_lax_friedrichs` at the moment since this is 
# `if`-clause free
surface_flux = flux_lax_friedrichs

# `RealT = Real` requires fewer overloads than the more explicit `RealT = Num` from Symbolics.
# `solver_real` is used for computing the Jacobian sparsity pattern
solver_real = DGSEM(polydeg = 3, surface_flux = surface_flux, RealT = Real)
# `solver_float` is  used for the subsequent simulation
solver_float = DGSEM(polydeg = 3, surface_flux = surface_flux)

###############################################################################
### mesh ###

# Mapping as described in https://arxiv.org/abs/2012.12040,
# reduced to 2D on [0, 2]^2 instead of [0, 3]^3
function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0,2]
    xi = xi_ + 1
    eta = eta_ + 1

    y = eta + 1 / 4 * (cos(pi * (xi - 1)) *
                       cos(0.5 * pi * (eta - 1)))

    x = xi + 1 / 4 * (cos(0.5 * pi * (xi - 1)) *
                      cos(2 * pi * (y - 1)))

    return SVector(x, y)
end
cells_per_dimension = (16, 16)
mesh = StructuredMesh(cells_per_dimension, mapping)

###############################################################################
### semidiscretizations ###

initial_condition = initial_condition_convergence_test

# `semi_real` is used for computing the Jacobian sparsity pattern
semi_real = SemidiscretizationHyperbolic(mesh, equations,
                                         initial_condition_convergence_test,
                                         solver_real,
                                         source_terms = source_terms_convergence_test)
# `semi_float` is  used for the subsequent simulation
semi_float = SemidiscretizationHyperbolic(mesh, equations,
                                          initial_condition_convergence_test,
                                          solver_float,
                                          source_terms = source_terms_convergence_test)

t0 = 0.0 # Re-used for the ODE function defined below
t_end = 5.0
t_span = (t0, t_end)

# Call `semidiscretize` on `semi_float` to create the ODE problem to have access to the initial condition.
ode_float = semidiscretize(semi_float, t_span)
u0_ode = ode_float.u0
du_ode = similar(u0_ode)

###############################################################################
### Compute the Jacobian with SparseDiffTools ###

# Create a function with two parameters: `du_ode` and `u0_ode`
# to fulfill the requirments of an in_place function in SparseDiffTools
# (see example function `f` from https://docs.sciml.ai/SparseDiffTools/dev/#Example)
rhs = (du_ode, u0_ode) -> Trixi.rhs!(du_ode, u0_ode, semi_real, t0)

# Taken from example linked above to detect the pattern and choose how to do the differentiation
sd = SymbolicsSparsityDetection()
ad_type = AutoForwardDiff()
sparse_adtype = AutoSparse(ad_type)

# `sparse_cache` will reduce calculation time when Jacobian is calculated multiple times
sparse_cache = sparse_jacobian_cache(sparse_adtype, sd, rhs, du_ode, u0_ode)

###############################################################################
### Set up sparse-aware ODEProblem ###

# Supply Jacobian prototype and coloring vector to the semidiscretization
ode_float_jac_sparse = semidiscretize(semi_float, t_span,
                                      sparse_cache.jac_prototype,
                                      sparse_cache.coloring.colorvec)

analysis_callback = AnalysisCallback(semi_float, interval = 50)
alive_callback = AliveCallback(alive_interval = 3)
summary_callback = SummaryCallback()

# Note: No `stepsize_callback` due to implicit solver
callbacks = CallbackSet(analysis_callback, alive_callback, summary_callback)

###############################################################################
### solve the ODE problem ###

sol = solve(ode_float_jac_sparse, # using `ode_float` is essentially infeasible, even single step takes ages!
            # `AutoForwardDiff()` is not yet working, probably related to https://docs.sciml.ai/DiffEqDocs/stable/basics/faq/#Autodifferentiation-and-Dual-Numbers
            Kvaerno4(; autodiff = AutoFiniteDiff());
            dt = 0.01, save_everystep = false, callback = callbacks);
