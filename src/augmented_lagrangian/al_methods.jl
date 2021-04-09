"""
al_methods.jl
"""

function solve!(solver::AugmentedLagrangianSolver{T,S}) where {T,S}
    # pull vars
    solver_uncon = solver.solver_uncon
    cost_tol = solver.opts.cost_tolerance
    grad_tol = solver.opts.gradient_tolerance
    c_max = Inf
    J = 0.
    # intialize
    set_verbosity!(solver)
    clear_cache!(solver)
    reset!(solver)
    # run AL iterations
    for i = 1:solver.opts.iterations_outer
	set_tolerances!(solver, solver_uncon, i, cost_tol, grad_tol)
        # solve the unconstrained problem
        J = solve!(solver.solver_uncon)
        # check solver status
        status(solver) > SOLVE_SUCCEEDED && break
        # record the updated information
        record_iteration!(solver, J, max_violation_penalty(solver)...)
        # check for convergence before doing the outer loop udpate
        converged = evaluate_convergence(solver)
        if converged
            break
        end
        # outer loop update
        dual_update!(solver)
        penalty_update!(solver)
        # reset verbosity level after it's modified
        set_verbosity!(solver)
        reset!(solver_uncon)
        if i == solver.opts.iterations_outer
            solver.stats.status = MAX_ITERATIONS
        end
    end
    solver.opts.cost_tolerance = cost_tol
    solver.opts.gradient_tolerance = grad_tol
    terminate!(solver)
    return J
end

function record_iteration!(solver::AugmentedLagrangianSolver{T,S}, J::T, c_max::T, μ_max::T) where {T,S}
    # Just update constraint violation and max penalty
    record_iteration!(solver.stats, c_max=c_max, penalty_max=μ_max, is_outer=true)
    j = solver.stats.iterations_outer::Int
    @logmsg OuterLoop :iter value=j
    @logmsg OuterLoop :total value=solver.stats.iterations
    @logmsg OuterLoop :cost value=J
    @logmsg OuterLoop :c_max value=c_max
    if is_verbose(solver) 
	print_level(OuterLoop, global_logger())
    end
    return nothing
end

function set_tolerances!(solver::AugmentedLagrangianSolver{T},
                         solver_uncon::AbstractSolver{T}, i::Int, 
                         cost_tol=solver.opts.cost_tolerance, 
                         grad_tol=solver.opts.gradient_tolerance) where T
    if i != solver.opts.iterations_outer
        solver_uncon.opts.cost_tolerance = solver.opts.cost_tolerance_intermediate
        solver_uncon.opts.gradient_tolerance = solver.opts.gradient_tolerance_intermediate
    else
        solver_uncon.opts.cost_tolerance = cost_tol 
        solver_uncon.opts.gradient_tolerance = grad_tol 
    end

    return nothing
end

function evaluate_convergence(solver::AugmentedLagrangianSolver)
    i = solver.stats.iterations
    converged = (solver.stats.c_max[i] < solver.opts.constraint_tolerance ||
	         solver.stats.penalty_max[i] >= solver.opts.penalty_max)
    return converged
end

