"""
al_methods.jl
"""

function solve!(solver::AugmentedLagrangianSolver{T,S}) where {T,S}
    # pull vars
    convals = solver.solver_uncon.obj.convals
    solver_uncon = solver.solver_uncon
    c_max = Inf
    J = 0.
    # intialize
    set_verbosity!(solver)
    clear_cache!(solver)
    reset!(solver)
    # run AL iterations
    for i = 1:solver.opts.al_max_iterations
        # solve the unconstrained problem
        J = solve!(solver.solver_uncon)
        # check solver status
        status(solver) > SOLVE_SUCCEEDED && break
        # record the updated information
        record_iteration!(solver, J, TO.max_violation_penalty(convals)...)
        # check for convergence before doing the outer loop udpate
        if (solver.stats.c_max[solver.stats.iterations] < solver.opts.al_vtol ||
	        solver.stats.penalty_max[solver.stats.iterations] >= solver.opts.penalty_max)
            break
        end
        # outer loop update
        TO.update_dual_penalty!(convals)
        # reset verbosity level after it's modified
        set_verbosity!(solver)
        reset!(solver_uncon)
        if i == solver.opts.al_max_iterations
            solver.stats.status = MAX_ITERATIONS_OUTER
        end
    end
    terminate!(solver)
    return J
end

function record_iteration!(solver::AugmentedLagrangianSolver{T,S},
                           J::T, c_max::T, μ_max::T) where {T,S}
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
