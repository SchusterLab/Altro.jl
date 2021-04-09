"""
al_solver.jl
"""

@doc raw""" ```julia
struct AugmentedLagrangianSolver <: ConstrainedSolver{T}
```
Augmented Lagrangian (AL) is a standard tool for constrained optimization. For a trajectory optimization problem of the form:
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
           & + \sum_{k=0}^{N-1} \ell_k(x_k,u_k,dt) + λ_k^T c_k(x_k,u_k) + c_k(x_k,u_k)^T I_{\mu_k} c_k(x_k,u_k)
\end{aligned}
```
This function is then minimized with respect to the primal variables using any unconstrained minimization solver (e.g. iLQR).
    After a local minima is found, the AL method updates the Lagrange multipliers λ and the penalty terms μ and repeats the unconstrained minimization.
    AL methods have superlinear convergence as long as the penalty term μ is updated each iteration.
"""
struct AugmentedLagrangianSolver{T,S<:AbstractSolver} <: ConstrainedSolver{T}
    opts::SolverOptions{T}
    stats::SolverStats{T}
    solver_uncon::S
end


"""$(TYPEDSIGNATURES)
Form an augmented Lagrangian cost function from a Problem and AugmentedLagrangianSolver.
    Does not allocate new memory for the internal arrays, but points to the arrays in the solver.
"""
function AugmentedLagrangianSolver(
        prob::Problem{IR}, 
        opts::SolverOptions=SolverOptions(), 
        stats::SolverStats=SolverStats(parent=solvername(AugmentedLagrangianSolver));
        solver_uncon=iLQRSolver,
        kwarg_opts...
    ) where {IR}
    set_options!(opts; kwarg_opts...)
    # set up al objective
    al_obj = ALObjective(prob, opts)
    # set up unconstrained solver
    prob_al = Problem(IR, prob.model, al_obj, ConstraintList(),
                      prob.X, prob.U, prob.ts, prob.N, prob.M, prob.V)
    solver_uncon = solver_uncon(prob_al, opts, stats)
    # set up al solver
    T = Float64
    S = typeof(solver_uncon)
    solver = AugmentedLagrangianSolver{T,S}(opts, stats, solver_uncon)
    reset!(solver)
    return solver
end

# methods
Base.size(solver::AugmentedLagrangianSolver) = size(solver.solver_uncon)
@inline TO.cost(solver::AugmentedLagrangianSolver) = TO.cost(solver.solver_uncon)
@inline TO.get_trajectory(solver::AugmentedLagrangianSolver) = get_trajectory(solver.solver_uncon)
@inline TO.get_objective(solver::AugmentedLagrangianSolver) = get_objective(solver.solver_uncon)
@inline TO.get_model(solver::AugmentedLagrangianSolver) = get_model(solver.solver_uncon)
@inline get_initial_state(solver::AugmentedLagrangianSolver) = get_initial_state(solver.solver_uncon)
@inline TrajectoryOptimization.integration(solver::AugmentedLagrangianSolver) = integration(solver.solver_uncon)
solvername(::Type{<:AugmentedLagrangianSolver}) = :AugmentedLagrangian
@inline TO.get_constraints(solver::AugmentedLagrangianSolver) = get_constraints(solver.solver_uncon)
@inline TO.states(solver::AugmentedLagrangianSolver) = TO.states(solver.solver_uncon)
@inline TO.controls(solver::AugmentedLagrangianSolver) = TO.controls(solver.solver_uncon)

function dual_penalty_update!(solver::AugmentedLagrangianSolver)
    for convals in solver.solver_uncon.obj.convals
        for conval in convals
            dual_update!(conval)
            penalty_update!(conval)
        end
    end
end

function max_violation_penalty(solver::AugmentedLagrangianSolver)
    max_violation = 0.
    max_penalty = 0.
    for (k, convals) in enumerate(solver.solver_uncon.obj.convals)
        for conval in convals
            viol = TO.violation(conval)
            max_violation = max(max_violation, viol)
            max_penalty = max(max_penalty, maximum(conval.μ))
        end
    end
    return max_violation, max_penalty
end

# options methods
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
    # reset stats
    reset!(solver.stats, solver.opts.iterations, :AugmentedLagrangian)
    # reset duals and penalties
    for convals in solver.solver_uncon.obj.convals
        for conval in convals
            conval.λ .= 0
            conval.μ .= conval.params.μ0
        end
    end
    # reset unconstrained solver
    reset!(solver.solver_uncon)
    return nothing
end
