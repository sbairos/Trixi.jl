using Trixi

###############################################################################
# create a restart file

elixir_file = "elixir_advection_implicit_sparse_jacobian.jl"
restart_file = "restart_000000006.h5"

trixi_include(@__MODULE__, joinpath(@__DIR__, elixir_file))

###############################################################################

restart_filename = joinpath("out", restart_file)
t_span = (load_time(restart_filename), 2.0)
dt_restart = load_dt(restart_filename)

ode_float_jac_sparse = semidiscretize(semi_float, t_span,
                                      sparse_cache.jac_prototype,
                                      sparse_cache.coloring.colorvec)

###############################################################################
# run the simulation

sol = solve(ode_float_jac_sparse, # using `ode_float_jac_sparse` instead of `ode_float` results in speedup of factors 10-15!
            # `AutoForwardDiff()` is not yet working, probably related to https://docs.sciml.ai/DiffEqDocs/stable/basics/faq/#Autodifferentiation-and-Dual-Numbers
            TRBDF2(; autodiff = ad_type); # `AutoForwardDiff()` is not yet working
            adaptive = true, dt = dt_restart, save_everystep = false, callback = callbacks);
