"""
pn_methods.jl
"""

function solve!(solver::ProjectedNewtonSolver)
    # initialize
    if solver.opts.verbose_pn
        println("\nProjection:")
    end
    ϵ_feas = solver.opts.pn_vtol
    # compute value and jacobian of all constraints
    evaluate_copy_constraints!(solver, solver.X, solver.U)
    viol = norm(solver.d[solver.a], Inf)
    # run solve
    step_count = 0
    while (step_count < solver.opts.n_steps) && (viol > ϵ_feas)
        viol = _projection_solve!(solver)
        if solver.opts.multiplier_projection
            res = multiplier_projection!(solver)
        else
            res = Inf
        end
        step_count += 1
        # compute cost
        J = 0.
        for k = 1:solver.N
            J += TO.cost(solver.obj, solver.X, solver.U, k)
        end
        J_prev = solver.stats.cost[solver.stats.iterations]
        dJ = J_prev - J
        # log
        record_iteration!(solver.stats, cost=J, c_max=viol, is_pn=true,
                          dJ=dJ, gradient=res, penalty_max=NaN)
    end
    # terminate
    terminate!(solver)
    return solver
end

function _projection_solve!(solver::ProjectedNewtonSolver)
    # initialize
    max_refinements = 10
    convergence_rate_threshold = solver.opts.r_threshold
    ρ_chol = solver.opts.ρ_chol
    ρ_primal = solver.opts.ρ_primal
    # update constraints and derivatives c, D, d, λ, a
    evaluate_copy_constraints!(solver, solver.X, solver.U)
    D = solver.D[solver.a, :]
    d = solver.d[solver.a]
    # update costs and derivatives H, g
    evaluate_copy_costs!(solver)
    H = Diagonal(solver.H)
    # check violation
    viol0 = norm(d, Inf)
    if solver.opts.verbose_pn
        println("feas0: $viol0")
    end
    # regularize Hessian
    if ρ_primal > 0.0
        for i = 1:solver.N
            H[i, i] += ρ_primal
        end
    end
    # compute projection operators
    HinvD = H \ D'
    S = Symmetric(D * HinvD)
    Sreg = cholesky(S + ρ_chol * I)
    # line search until convergence
    viol_prev = viol0
    refine_count = 0
    while refine_count < max_refinements
        # line search
        viol = _projection_linesearch!(solver, (S, Sreg), HinvD)
        # log conv rate
        convergence_rate = log10(viol) / log10(viol_prev)
        viol_prev = viol
        if solver.opts.verbose_pn
            println("conv rate: $convergence_rate")
        end
        # exit if converged
        if ((convergence_rate < convergence_rate_threshold) ||
            (viol < solver.opts.pn_vtol))
            break
        end
        refine_count += 1
    end
    return viol_prev
end

function _projection_linesearch!(solver::ProjectedNewtonSolver, S, HinvD)
    # initialize
    viol = Inf
    solve_tol = 1e-8
    refinement_iters = 25
    α = 1.
    ϕ = 0.5
    viol_decreased = false
    max_iter_count = 10
    # grab variables
    N = solver.N
    convals = solver.convals
    a = solver.a
    d = solver.d[a]
    viol0 = norm(d, Inf)
    X = solver.X
    X_tmp = solver.X_tmp
    x_ginds = solver.x_ginds
    U = solver.U
    U_tmp = solver.U_tmp
    u_ginds = solver.u_ginds
    # run linesearch until violation decreases or
    # maximum number of searches
    count = 1
    while true
        δλ = reg_solve(S[1], d, S[2], solve_tol, refinement_iters)
        δZ = -HinvD * δλ
        δZ .*= α
        # build temporary trajectory from update
        for k = 1:N - 1
            solver.X_tmp[k] .= solver.X[k]
            solver.X_tmp[k] .+= δZ[x_ginds[k]]
            solver.U_tmp[k] .= solver.U[k]
            solver.U_tmp[k] .+= δZ[u_ginds[k]]
        end
        solver.X_tmp[N] .= solver.X[N]
        solver.X_tmp[N] .+= δZ[x_ginds[N]]
        # check constraint violation
        evaluate_copy_constraints!(solver, X_tmp, U_tmp)
        max_violation, _ = TO.max_violation_penalty(convals)
        d = solver.d[a]
        viol = norm(d, Inf)
        # log
        if solver.opts.verbose_pn
            println("feas: $(viol) (α = $(α))")
        end
        # evaluate convergence
        if viol < viol0
            viol_decreased = true
            break
        elseif count > max_iter_count
            break
        end
        count += 1
        α *= ϕ
    end
    # accept updated trajectory if the violation decreased
    if viol_decreased
        for k = 1:solver.N - 1
            solver.X[k] .= solver.X_tmp[k]
            solver.U[k] .= solver.U_tmp[k]
        end
        solver.X[N] .= solver.X_tmp[N]
    end
    return viol
end

function multiplier_projection!(solver::ProjectedNewtonSolver)
    D = solver.D[solver.a, :]
    d = solver.d[solver.a]
    λ = solver.λ[solver.a]
    g = solver.g
    res0 = g + D'λ
    A = D * D'
    Areg = A + I * solver.opts.ρ_primal
    b = D * res0
    δλ = -reg_solve(A, b, Areg)
    λ += δλ
    res = g + D'λ  # primal residual
    return norm(res)
end

function reg_solve(A, b, B, tol=1e-10, max_iters=10)
    x = B \ b
    count = 0
    while count < max_iters
        r = b - A * x
        if norm(r) < tol
            break
        else
            x += B \ r
            count += 1
        end
    end
    return x
end

"""
compute the value and jacobian of all constraints along the specified
trajectory, storing them in the proper linearized constraint
"""
function evaluate_copy_constraints!(solver::ProjectedNewtonSolver,
                                    X::Vector{<:AbstractVector},
                                    U::Vector{<:AbstractVector}; eval=true, jac=true,
                                    override_const_jac=false)
    for k = 1:solver.N
        for conval in solver.convals[k]
            # evaluate, active, copy
            if eval
                TO.evaluate!(conval.c, conval.con, X, U, k)
                TO.update_active!(conval.a, conval.con, conval.c, conval.λ, 0.)
                solver.d[conval.c_ginds] .= conval.c
                solver.λ[conval.c_ginds] .= conval.λ
                solver.a[conval.c_ginds] .= conval.a
            end
            # jacobian, copy
            if jac && !conval.con.const_jac
                TO.jacobian_copy!(solver.D, conval.con, X, U, k, conval.c_ginds, solver.x_ginds,
                                  solver.u_ginds)
            end
        end
    end
    return nothing
end

function evaluate_copy_costs!(solver::ProjectedNewtonSolver)
    H = solver.H
    g = solver.g
    E = solver.E
    x_ginds = solver.x_ginds
    u_ginds = solver.u_ginds
    N = solver.N
    for k = 1:N-1
        TO.cost_derivatives!(E, solver.obj, solver.X, solver.U, k)
        H[x_ginds[k], x_ginds[k]] .= E.Q
        H[u_ginds[k], u_ginds[k]] .= E.R
        H[u_ginds[k], x_ginds[k]] .= E.H
        g[x_ginds[k]] .= E.q
        g[u_ginds[k]] .= E.r
    end
    TO.cost_derivatives!(E, solver.obj, solver.X, solver.U, N)
    H[x_ginds[N], x_ginds[N]] .= E.Q
    g[x_ginds[N]] .= E.q
end
