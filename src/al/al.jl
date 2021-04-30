"""
al_solver.jl
"""

import Base: show

@doc raw""" ```julia
struct AugmentedLagrangianSolver <: ConstrainedSolver{T}
```
Augmented Lagrangian (AL) is a standard tool for constrained optimization.
For a trajectory optimization problem of the form:
```math
\begin{aligned}
  \min_{x_{0:N},u_{0:N-1}} \quad & \ell_f(x_N) + \sum_{k=0}^{N-1} \ell_k(x_k, u_k, dt) \\
  \textrm{s.t.}            \quad & x_{k+1} = f(x_k, u_k), \\
                                 & g_k(x_k,u_k) \leq 0, \\
                                 & h_k(x_k,u_k) = 0.
\end{aligned}
```
AL methods form the following augmented Lagrangian function:
```math
\begin{aligned}
    \ell_f(x_N) + &λ_N^T c_N(x_N) + c_N(x_N)^T I_{\mu_N} c_N(x_N) \\
           & + \sum_{k=0}^{N-1} \ell_k(x_k,u_k,dt) + λ_k^T c_k(x_k,u_k)
+ c_k(x_k,u_k)^T I_{\mu_k} c_k(x_k,u_k)
\end{aligned}
```
This function is then minimized with respect to the primal variables using any
unconstrained minimization solver (e.g. iLQR).
After a local minima is found, the AL method updates the Lagrange multipliers λ
and the penalty terms μ and repeats the unconstrained minimization.
AL methods have superlinear convergence as long as the penalty term μ
is updated each iteration.
"""
struct AugmentedLagrangianSolver{T,S} <: ConstrainedSolver{T}
    opts::SolverOptions{T}
    stats::SolverStats{T}
    solver_uncon::S
    solver_name::Symbol
end


"""$(TYPEDSIGNATURES)
Form an augmented Lagrangian cost function from a Problem and AugmentedLagrangianSolver.
Does not allocate new memory for the internal arrays, but points to the arrays in the solver.
"""
function AugmentedLagrangianSolver(prob::Problem{IR,T}, opts::SolverOptions,
                                   stats::SolverStats; solver_uncon=iLQRSolver) where {IR,T}
    # set up unconstrained solver
    solver_uncon = solver_uncon(prob, opts, stats)
    # put it all together
    S = typeof(solver_uncon)
    solver = AugmentedLagrangianSolver{T,S}(opts, stats, solver_uncon, :AugmentedLagrangian)
    reset!(solver)
    return solver
end

# options methods
Base.show(io::IO, solver::AugmentedLagrangianSolver) = (
    print(io, "ALSolver")
)

Base.show(io::IO, m::MIME"text/plain", solver::AugmentedLagrangianSolver) = (
    print(io, "ALSolver")
)

function set_verbosity!(solver::AugmentedLagrangianSolver)
    llevel = log_level(solver) 
    if is_verbose(solver)
        set_logger()
        Logging.disable_logging(LogLevel(llevel.level-1))
        logger = global_logger()
        if is_verbose(solver.solver_uncon) 
            freq = 1
        else
            freq = 5
        end
        logger.leveldata[llevel].freq = freq
    else
        Logging.disable_logging(llevel)
    end
end


function reset!(solver::AugmentedLagrangianSolver)
    # reset duals and penalties
    for convals in solver.solver_uncon.obj.convals
        for conval in convals
            conval.params.ϕ = solver.opts.penalty_scaling
            conval.params.μ0 = solver.opts.penalty_initial
            conval.params.μ_max = solver.opts.penalty_max
            conval.params.λ_max = solver.opts.dual_max
            conval.λ .= 0
            conval.μ .= conval.params.μ0
            conval.a .= false
        end
    end
    # reset unconstrained solver
    reset!(solver.solver_uncon)
    return nothing
end
