using Trixi
using OrdinaryDiffEqSDIRK

# Functionality for automatic sparsity detection
using SparseDiffTools, Symbolics

import Base: eps, zero, one, * # For overloading with type `Real`
###############################################################################################
### Overloads to construct the `LobattoLegendreBasis` with `Real` type (supertype of `Num`) ###

# Required for setting up the Lobatto-Legendre basis for abstract `Real` type.
# Constructing the Lobatto-Legendre basis with `Real` instead of `Num` is 
# significantly easier as we do not have to care about e.g. if-clauses.
# As a consequence, we need to provide some overloads hinting towards the intended behavior.

const float_type = Float64 # Actual floating point type for the simulation

# Newton tolerance for finding LGL nodes & weights
Trixi.eps(::Type{Real}) = Base.eps(float_type)
# # There are some places where `one(RealT)` or `zero(uEltype)` is called where `RealT` or `uEltype` is `Real`.
# # This returns an `Int64`, i.e., `1` or `0`, respectively which gives errors when a floating-point alike type is expected.
Trixi.one(::Type{Real}) = Base.one(float_type)
# Trixi.zero(::Type{Real}) = Base.zero(float_type)

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
# semidiscretization of the linear advection diffusion equation

advection_velocity = (1.5, 1.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
diffusivity() = 5.0e-2
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
# `RealT = Real` requires fewer overloads than the more explicit `RealT = Num` from Symbolics
# `solver_real` is used for computing the Jacobian sparsity pattern
solver_real = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs, RealT = Real)
# `solver_float` is  used for the subsequent simulation
solver_float = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000, # set maximum capacity of tree data structure
                periodicity = true)

# Define initial condition
function initial_condition_diffusive_convergence_test(x, t,
                                                      equation::LinearScalarAdvectionEquation2D)
    # Store translated coordinate for easy use of exact solution
    RealT = eltype(x)
    x_trans = x - equation.advection_velocity * t

    nu = diffusivity()
    c = 1
    A = 0.5f0
    L = 2
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f
    scalar = c + A * sin(omega * sum(x_trans)) * exp(-2 * nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

# A semidiscretization collects data structures and functions for the spatial discretization
# `semi_real` is used for computing the Jacobian sparsity pattern
semi_real = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition,
                                             solver_real;
                                             solver_parabolic = ViscousFormulationBassiRebay1(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))
# `semi_float` is  used for the subsequent simulation
semi_float = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition,
                                             solver_float;
                                             solver_parabolic = ViscousFormulationBassiRebay1(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

                                                                    
# Create ODE problem with time span from 0.0 to 1.0
t0 = 0.0 # Re-used for the ODE function defined below
tspan = (t0, 1.5)
ode_float = semidiscretize(semi_float, tspan)
u0_ode = ode_float.u0
du_ode = similar(u0_ode)

###############################################################################################
### Compute the Jacobian with SparseDiffTools ###

# Create a function with two parameters: `du_ode` and `u0_ode`
# to fulfill the requirements of an in_place function in SparseDiffTools
# (see example function `f` from https://docs.sciml.ai/SparseDiffTools/dev/#Example)

function rhs_hyperbolic_parabolic!(du_ode, u_ode)
    Trixi.@trixi_timeit Trixi.timer() "rhs_hyperbolic_parabolic!" begin
        # Implementation of split ODE problem in OrdinaryDiffEq
        du_para = similar(du_ode) # This obviously allocates
        Trixi.rhs!(du_ode, u_ode, semi_real, t0) # hyperbolic part
        Trixi.rhs_parabolic!(du_para, u_ode, semi_real, t0)

        Trixi.@threaded for i in eachindex(du_ode)
            # Try to enable optimizations due to `muladd` by avoiding `+=`
            # https://github.com/trixi-framework/Trixi.jl/pull/2480#discussion_r2224531702
            du_ode[i] = du_ode[i] + du_para[i]
        end
    end
end
# rhs = (du_ode, u0_ode) -> Trixi.rhs!(du_ode, u0_ode, semi_real, t0)
# rhs_parabolic = (du_para, u0_para) -> Trixi.rhs_parabolic!(du_para, u0_para, semi_real, t0)

# Taken from example linked above to detect the pattern and choose how to do the AutoDiff automatically
sd = SymbolicsSparsityDetection()
ad_type = AutoFiniteDiff()
sparse_adtype = AutoSparse(ad_type)

# `sparse_cache` will reduce calculation time when Jacobian is calculated multiple times,
# which is in principle not required for the linear problem considered here.
sparse_cache = sparse_jacobian_cache(sparse_adtype, sd, rhs_hyperbolic_parabolic!, du_ode, u0_ode)
const jac_sparse = sparse_jacobian(ad_type, sparse_cache, rhs_hyperbolic_parabolic!, du_ode, u0_ode)
const jac_sparse_func!(J, u, p, t) = jac_sparse
###############################################################################################
### Set up sparse-aware ODEProblem ###

# Revert overrides from above for the actual simulation
Trixi.eps(x::Type{Real}) = Base.eps(x)
Trixi.one(x::Type{Real}) = Base.one(x)
# Trixi.zero(x::Type{Real}) = Base.zero(x)

# Supply Jacobian prototype and coloring vector to the semidiscretization
ode_float_jac_sparse2 = Trixi.semidiscretize2(semi_float, tspan,
                                      jac_shared = jac_sparse_func!#,
                                    #   sparse_cache_para.jac_prototype,
                                    #   sparse_cache_para.coloring.colorvec
                                      )

# Note: We experimented for linear problems with providing the constant, sparse Jacobian directly via
#
# const jac_sparse = sparse_jacobian(sparse_adtype, sparse_cache, rhs, du_ode, u0_ode)
# const jac_sparse_func!(J, u, p, t) = jac_sparse
# SciMLBase.ODEFunction(rhs!, jac_prototype=float.(jac_prototype), colorvec=colorvec, jac = jac_sparse_func!)
#
# which turned out to be significantly slower than just using the prototype and the coloring vector. 

###############################################################################
# ODE solvers, callbacks etc.

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi_float, interval = 100)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval = 100)

# The SaveRestartCallback allows to save a file from which a Trixi.jl simulation can be restarted
save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_restart)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
time_int_tol = 1.0e-9
time_abs_tol = 1.0e-9

sol2 = solve(ode_float_jac_sparse2, TRBDF2(; autodiff = ad_type);
            adaptive = true, save_everystep = false,
            abstol = time_abs_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
