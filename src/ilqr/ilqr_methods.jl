"""
ilqr_solve.jl
"""

"iLQR solve method"
function solve!(solver::iLQRSolver{IR}) where {IR}
    # initialize
    reset!(solver)
    set_verbosity!(solver)
    clear_cache!(solver)
    solver.ρ[1] = solver.opts.bp_reg_initial
    solver.dρ[1] = 0.0
    n, m, N = size(solver)
    X = solver.X
    U = solver.U
    ts = solver.ts

    # initial rollout and cost calculation
    J = J_prev = 0.
    for k = 1:N-1
        J_prev += cost(solver.obj, X, U, k)
        dt = ts[k + 1] - ts[k]
        discrete_dynamics!(X[k + 1], IR, solver.model, X[k], U[k], ts[k], dt)
    end
    J_prev += cost(solver.obj, X, U, N)

    # run iLQR iterations
    for i = 1:solver.opts.ilqr_max_iterations
        # step
	    backwardpass!(solver)
        J, reg_flag = forwardpass!(solver, J_prev)
        # exit if solve succeeded
        if solver.stats.status > SOLVE_SUCCEEDED
            break
        end
        # accept the updated trajectory
        for k = 1:N-1
            solver.X[k] .= solver.X_tmp[k]
            solver.U[k] .= solver.U_tmp[k]
        end
        solver.X[N] .= solver.X_tmp[N]
        # record iteration
        dJ = abs(J - J_prev)
        J_prev = J
        gradient_todorov!(solver)
        record_iteration!(solver, J, dJ)
        if is_verbose(solver) 
            print_level(InnerLoop, global_logger())
        end
        # exit if converged
        exit = evaluate_convergence(solver, i, reg_flag)
        exit && break
    end
    terminate!(solver)
    return J
end


"""
$(SIGNATURES)
Simulate the system forward using the optimal feedback gains from the backward pass,
projecting the system on the dynamically feasible subspace. Performs a line search to ensure
adequate progress on the nonlinear problem.
"""
function forwardpass!(solver::iLQRSolver, J_prev)
    ΔV = solver.ΔV
    J = Inf
    α = 1.0
    iter = 0
    z = -1.0
    expected = 0.0
    rollout_flag = false
    reg_flag = false

    while ((z ≤ solver.opts.line_search_lower_bound
            || z > solver.opts.line_search_upper_bound)
           && J >= J_prev)
        # exit and regularize if the maximum number of line search decrements
        # has occured
        if iter > solver.opts.iterations_linesearch
            J = J_prev
            z = 0
            α = 0.0
            expected = 0.0
            regularization_update!(solver, :increase)
            solver.ρ[1] += solver.opts.bp_reg_fp
            reg_flag = true
            break
        end
        # otherwise, rollout a new trajectory for current alpha
        J, rollout_flag = rollout!(solver, α)
        # reduce step size if rollout returns non-finite values (NaN or Inf)
        if rollout_flag
            J = J_prev
            iter += 1
            α /= 2.0
            continue
        end
        # update
        expected = -α * (ΔV[1] + α * ΔV[2])
        if expected > 0.0
            z  = (J_prev - J)/expected
        else
            z = -1.0
        end
        iter += 1
        α /= 2.0
    end

    if J > J_prev
        # error("Error: Cost increased during Forward Pass")
        solver.stats.status = COST_INCREASE
        return NaN
    end

    @logmsg InnerLoop :expected value=expected
    @logmsg InnerLoop :z value=z
    @logmsg InnerLoop :α value=2*α
    @logmsg InnerLoop :ρ value=solver.ρ[1]

    return J, reg_flag
end


"""
Stash iteration statistics
"""
function record_iteration!(solver::iLQRSolver, J, dJ)
    gradient = mean(solver.grad)
    record_iteration!(solver.stats, cost=J, dJ=dJ, gradient=gradient)
    i = solver.stats.iterations::Int
    
    if dJ ≈ 0
        solver.stats.dJ_zero_counter += 1
    else
        solver.stats.dJ_zero_counter = 0
    end

    @logmsg InnerLoop :iter value=i
    @logmsg InnerLoop :cost value=J
    @logmsg InnerLoop :dJ   value=dJ
    @logmsg InnerLoop :grad value=gradient
    # @logmsg InnerLoop :zero_count value=solver.stats[:dJ_zero_counter][end]
    return nothing
end

"""
$(SIGNATURES)
    Calculate the problem gradient using heuristic from iLQG (Todorov) solver
"""
function gradient_todorov!(solver::iLQRSolver)
    N = solver.N
    tmp = solver.U_tmp[N]
    u = solver.U_tmp[N+1]
    for k in eachindex(solver.d)
	tmp .= abs.(solver.d[k])
	u .= abs.(solver.U[k]) .+ 1
	tmp ./= u
	solver.grad[k] = maximum(tmp)
    end
end


"""
$(SIGNATURES)
Check convergence conditions for iLQR
"""
function evaluate_convergence(solver::iLQRSolver, ilqr_iterations::Int, reg_flag::Bool)
    # Get current iterations
    i = solver.stats.iterations
    grad = solver.stats.gradient[i]
    dJ = solver.stats.dJ[i]
    J = solver.stats.cost[i]

    # If the change in cost is small, the gradient is small, and
    # the last step was not a regularization step, we have converged
    if ((0.0 <= dJ < solver.opts.ilqr_ctol) && (grad < solver.opts.ilqr_gtol)
        && !reg_flag)
        @logmsg InnerLoop "Cost criteria satisfied."
        solver.stats.status = SOLVE_SUCCEEDED
        return true
    end

    # Check total iterations
    if i >= solver.opts.iterations
        @logmsg InnerLoop "Hit max iterations. Terminating."
        solver.stats.status = MAX_ITERATIONS
        return true
    end

    # Exit iLQR and continue solving if the maximum number of iLQR iterations
    # has been reached.
    if ilqr_iterations >= solver.opts.ilqr_max_iterations
        return true
    end

    # Outer loop update if forward pass is repeatedly unsuccessful
    if solver.stats.dJ_zero_counter > solver.opts.dJ_counter_limit
        @logmsg InnerLoop "dJ Counter hit max. Terminating."
        solver.stats.status = NO_PROGRESS
        return true
    end

    # Terminate if max cost value was reached.
    if J > solver.opts.max_cost_value
        @logmsg InnerLoop "Hit max cost value. Terminating."
        solver.stats.status = MAXIMUM_COST
        return true
    end

    return false
end

"""
$(SIGNATURES)
Update the regularzation for the iLQR backward pass
"""
function regularization_update!(solver::iLQRSolver,status::Symbol=:increase)
    # println("reg $(status)")
    if status == :increase # increase regularization
        # @logmsg InnerLoop "Regularization Increased"
        solver.dρ[1] = max(solver.dρ[1] * solver.opts.bp_reg_increase_factor,
                           solver.opts.bp_reg_increase_factor)
        solver.ρ[1] = max(solver.ρ[1] * solver.dρ[1], solver.opts.bp_reg_min)
        # if solver.ρ[1] > solver.opts.bp_reg_max
        #     @warn "Max regularization exceeded"
        # end
    elseif status == :decrease # decrease regularization
        # TODO: Avoid divides by storing the decrease factor (divides are 10x slower)
        solver.dρ[1] = min(solver.dρ[1] / solver.opts.bp_reg_increase_factor,
                           1. / solver.opts.bp_reg_increase_factor)
        solver.ρ[1] = solver.ρ[1] * solver.dρ[1] * (solver.ρ[1] * solver.dρ[1]
                                                    > solver.opts.bp_reg_min)
    end
end
