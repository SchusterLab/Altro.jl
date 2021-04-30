"""
    AbstractSolver{T} <: MathOptInterface.AbstractNLPEvaluator

Abstract solver for trajectory optimization problems

# Interface
Any type that inherits from `AbstractSolver` must define the following methods:
```julia
model = get_model(::AbstractSolver)::AbstractModel
obj   = get_objective(::AbstractSolver)::AbstractObjective
E     = get_cost_expansion(::AbstractSolver)::QuadraticExpansion  # quadratic error state expansion
Z     = get_trajectory(::AbstractSolver)::Traj
n,m,N = Base.size(::AbstractSolver)
x0    = get_initial_state(::AbstractSolver)::StaticVector
solve!(::AbstractSolver)
```

Optional methods for line search and merit function interface. Note that these do not
    have to return `Traj`
```julia
Z     = get_solution(::AbstractSolver)  # current solution (defaults to get_trajectory)
Z     = get_primals(::AbstractSolver)   # current primals estimate used in the line search
dZ    = get_step(::AbstractSolver)      # current step in the primal variables
```

Optional methods
```julia
opts  = options(::AbstractSolver)       # options struct for the solver. Defaults to `solver.opts`
st    = solver_stats(::AbstractSolver)  # dictionary of solver statistics. Defaults to `solver.stats`
iters = iterations(::AbstractSolver)    #
```
"""
abstract type AbstractSolver{T} end

"$(TYPEDEF) Unconstrained optimization solver. Will ignore
any constraints in the problem"
abstract type UnconstrainedSolver{T} <: AbstractSolver{T} end


"""$(TYPEDEF)
Abstract solver for constrained trajectory optimization problems

In addition to the methods required for `AbstractSolver`, all `ConstrainedSolver`s
    must define the following method
```julia
get_constraints(::ConstrainedSolver)::ConstrainSet
```
"""
abstract type ConstrainedSolver{T} <: AbstractSolver{T} end

# methods
@inline iterations(solver::AbstractSolver) = solver.stats.iterations
@inline status(solver::AbstractSolver) = solver.stats.status
@inline is_constrained(::Type{<:AbstractSolver})::Bool = true
@inline is_constrained(::Type{<:ConstrainedSolver})::Bool = true
@inline is_constrained(::Type{<:UnconstrainedSolver})::Bool = false
@inline is_constrained(solver::AbstractSolver) = is_constrained(typeof(solver)) && !isempty(get_constraints(solver))

"""
    terminate!(solver::AbstractSolver)

Perform any necessary actions after finishing the solve.
"""
function terminate!(solver::AbstractSolver)
    # Time solve
    stat = solver.stats
    stat.tsolve = (time_ns() - stat.tstart)*1e-6  # in ms

    # Delete extra stats entries, only if terminal solver
    trim!(solver.stats, solver.solver_name)

    # Print solve summary
    if solver.opts.show_summary && is_parentsolver(solver)
        print_summary(solver)
    end
end

"""
    TerminationStatus

* `UNSOLVED`: Initial value. Solve either hasn't been attempted or is in process.. 
* `SOLVE_SUCCEEDED`: Solve met all the required convergence criteria.
* `MAX_ITERATIONS`: Solve was unable to meet the required convergence criteria within the maximum number of iterations.
* `MAX_ITERATIONS_OUTER`: Solve was unable to meet the required constraint satisfaction the maximum number of outer loop iterations.
* `MAXIMUM_COST`: Cost exceeded maximum allowable cost.
* `STATE_LIMIT`: State values exceeded the imposed numerical limits.
* `CONTROL_LIMIT`: Control values exceeded the imposed numerical limits.
* `NO_PROGRESS`: iLQR was unable to make any progress for `dJ_counter_limit` consecutive iterations.
* `COST_INCREASE`: The cost increased during the iLQR forward pass.
"""
@enum TerminationStatus begin
    UNSOLVED = 1
    MAX_ITERATIONS_OUTER = 2
    MAX_ITERATIONS = 3
    SOLVE_SUCCEEDED = 4
    MAXIMUM_COST = 5
    STATE_LIMIT = 6
    CONTROL_LIMIT = 7
    NO_PROGRESS = 8
    COST_INCREASE = 9
end

function print_summary(solver::S) where S <: AbstractSolver
    stat = stats(solver)
    col_h1 = crayon"bold green"
    col_h2 = crayon"bold blue"
    col0 = Crayon(reset=true)
    get_color(v::Bool) = v ? crayon"green" : crayon"red" 

    # Info header
    println(col_h1, "\nSOLVE COMPLETED")
    print(col0," solved using the ")
    print(col0, crayon"bold cyan", solvername(solver))
    print(col0, " Solver,\n part of the Altro.jl package developed by the REx Lab at Stanford and Carnegie Mellon Universities\n")

    # Stats
    println(col_h2, "\n  Solve Statistics")
    println(col0, "    Total Iterations: ", iterations(solver))
    println(col0, "    Solve Time: ", stat.tsolve, " (ms)")

    # Convergence
    println(col_h2, "\n  Covergence")
    if iterations(solver) == 0
        println(crayon"red", "    Solver failed to make it through the first iteration.")
    else
        println(col0, "    Terminal Cost: ", stat.cost[end])
        println(col0, "    Terminal dJ: ", get_color(stat.dJ[end] < solver.opts.cost_tolerance), stat.dJ[end])
        println(col0, "    Terminal gradient: ", get_color(stat.gradient[end] < solver.opts.gradient_tolerance), stat.gradient[end])
        if is_constrained(solver)
            println(col0, "    Terminal constraint violation: ", get_color(stat.c_max[end] < solver.opts.constraint_tolerance), stat.c_max[end])
        end
    end
    println(col0, "    Solve Status: ", crayon"bold", get_color(status(solver) == SOLVE_SUCCEEDED), status(solver))
    print(Crayon(reset=true))  # reset output color
end

# logging

log_level(solver::AbstractSolver) = OuterLoop

is_verbose(solver::AbstractSolver) = 
    log_level(solver) >= LogLevel(-100*solver.opts.verbose)

function set_verbosity!(solver::AbstractSolver)
    llevel = log_level(solver)
    if is_verbose(solver)
        set_logger()
        Logging.disable_logging(LogLevel(llevel.level-1))
    else
        Logging.disable_logging(llevel)
    end
end

function clear_cache!(solver::AbstractSolver)
    llevel = log_level(solver)
    if is_verbose(solver)
        SolverLogging.clear_cache!(global_logger().leveldata[llevel])
    end
end
