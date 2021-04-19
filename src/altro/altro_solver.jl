"""
altro_solver.jl
"""

"""$(TYPEDEF)
Augmented Lagrangian Trajectory Optimizer (ALTRO) is a solver developed by
the Robotic Exploration Lab at Stanford University.
The solver is special-cased to solve Markov Decision Processes
by leveraging the internal problem structure.

ALTRO consists of two "phases":
1) AL-iLQR: iLQR is used with an Augmented Lagrangian framework to solve the
problem quickly to rough constraint satisfaction
2) Projected Newton: A collocation-flavored active-set solver projects the solution
from AL-iLQR onto the feasible subspace to achieve machine-precision constraint satisfaction.
"""
struct ALTROSolver{T,S,Tp} <: ConstrainedSolver{T}
    opts::SolverOptions{T}
    stats::SolverStats{T}
    solver_al::AugmentedLagrangianSolver{T,S}
    solver_pn::Tp
    solver_name::Symbol
end

function ALTROSolver(prob::Problem{IR,T,Tm,<:Any,Tx,Tix,Tu,Tiu,Tt,TE,TM,TMd,TV},
                     opts::SolverOptions;
                     solver_uncon=iLQRSolver) where {IR,T,Tm,Tx,Tix,Tu,Tiu,Tt,TE,TM,TMd,TV}
    stats = SolverStats{T}(parent=:ALTRO)
    # make a new problem that references the old, but has ALObjective
    al_obj = ALObjective(prob, opts)
    To = typeof(al_obj)
    prob_al = Problem{IR,T,Tm,To,Tx,Tix,Tu,Tiu,Tt,TE,TM,TMd,TV}(
        prob.n, prob.m, prob.N, prob.model, al_obj, prob.convals, prob.X,
        prob.X_tmp, prob.ix, prob.U, prob.U_tmp, prob.iu, prob.ts, prob.E,
        prob.M, prob.Md, prob.V
    )
    # construct sub-solvers
    solver_al = AugmentedLagrangianSolver(prob_al, opts, stats; solver_uncon=solver_uncon)
    solver_pn = ProjectedNewtonSolver(prob_al, opts, stats)
    # put it all together
    S = typeof(solver_al.solver_uncon)
    Tp = typeof(solver_pn)
    solver = ALTROSolver{T,S,Tp}(opts, stats, solver_al, solver_pn, :ALTRO)
    return solver
end

# methods
@inline states(solver::ALTROSolver) = solver.solver_pn.X
@inline controls(solver::ALTROSolver) = solver.solver_pn.U
@inline max_violation_info(solver::ALTROSolver) = TO.max_violation_info(solver.solver_pn.convals)

function reset!(solver::ALTROSolver)
    reset!(solver.stats, solver.opts.iterations + 1, solver.solver_name)
    reset!(solver.solver_al)
end

# solve
function solve!(solver::ALTROSolver)
    reset!(solver)
    
    # solve with AL
    solve!(solver.solver_al)

    if status(solver) <= SOLVE_SUCCEEDED
        # Check convergence
        i = solver.solver_al.stats.iterations
        c_max = solver.solver_al.stats.c_max[i]
        if solver.opts.projected_newton && c_max > solver.opts.pn_vtol
            solve!(solver.solver_pn)
        end
        # Back-up check
        if status(solver) <= SOLVE_SUCCEEDED 
            # TODO: improve this check
            if TO.max_violation_penalty(solver.solver_pn.convals)[1] < solver.opts.pn_vtol
                solver.stats.status = SOLVE_SUCCEEDED
            end
        end
    end

    terminate!(solver)
    solver
end
