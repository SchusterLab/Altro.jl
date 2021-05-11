"""
problem.jl
"""

"""$(TYPEDEF) Trajectory Optimization Problem.
Contains the full definition of a trajectory optimization problem, including:
* dynamics model (`Model`)
* objective (`Objective`)
* constraints (`ConstraintSet`)
* initial and final states
* Primal variables (state and control trajectories)
* Discretization information: knot points (`N`), time step (`dt`), and total time (`tf`)

# Constructors:
```julia
Problem(model, obj, constraints, x0, xf, Z, N, tf) # defaults to RK3 integration
Problem{Q}(model, obj, constraints, x0, xf, Z, N, tf) where Q<:QuadratureRule
Problem(model, obj, xf, tf; x0, constraints, N, X0, U0, dt, integration)
Problem{Q}(prob::Problem)  # change integration
```
where `Z` is a trajectory (Vector of `KnotPoint`s)

# Arguments
* `model`: Dynamics model. Can be either `Discrete` or `Continuous`
* `obj`: Objective
* `X0`: Initial state trajectory. If omitted it will be initialized with NaNs, to be later overwritten by the solver.
* `U0`: Initial control trajectory. If omitted it will be initialized with zeros.
* `x0`: Initial state. Defaults to zeros.
* `xf`: Final state. Defaults to zeros.
* `dt`: Time step
* `tf`: Final time. Set to zero to specify a time penalized problem.
* `N`: Number of knot points. Defaults to 51, unless specified by `dt` and `tf`.
* `integration`: One of the defined integration types to discretize the continuous dynamics model.
Both `X0` and `U0` can be either a `Matrix` or a `Vector{Vector}`, but must be the same.
At least 2 of `dt`, `tf`, and `N` need to be specified (or just 1 of `dt` and `tf`).
"""
struct Problem{IR,T,Tm,To,Tx,Tix,Tu,Tiu,Tt,TE,TM,TMd,TV}
    # problem info
    n::Int
    m::Int
    N::Int
    model::Tm
    obj::To
    convals::Vector{Vector{ConVal}}
    X::Vector{Tx}
    X_tmp::Vector{Tx}
    ix::Tix
    U::Vector{Tu}
    U_tmp::Vector{Tu}
    iu::Tiu
    ts::Tt
    E::TE
    M::TM
    Md::TMd
    V::TV
end

function Problem(::Type{IR}, model::Tm, obj::To, constraints::ConstraintList,
                 X::Vector{Tx}, U::Vector{Tu}, ts::Tt, N::Int, M::TM, Md::TMd, V::TV) where {
                     IR<:QuadratureRule,Tm<:AbstractModel,To<:AbstractObjective,
                     Tx<:AbstractVector,Tu<:AbstractVector,Tt<:AbstractVector,
                     TM,TMd,TV}
    n, m = size(model)
    # allocate shared resources
    T = eltype(X[1])
    X_tmp = [V(zeros(T, n)) for k = 1:N+1] # 1 extra
    U_tmp = [V(zeros(T, m)) for k = 1:N+1] # 2 extra
    E = QuadraticCost(M(zeros(T, n, n)), M(zeros(T, m, m)), M(zeros(T, m, n)), V(zeros(T, n)),
                      V(zeros(T, m)), zero(T); checks=false)
    # construct indices into concatenated state and controls
    ix = V(1:n)
    iu = V((1:m) .+ n)
    # initial condition constraint for direct solve
    initial_state_constraint = GoalConstraint(copy(X[1]), V(1:n), n, m, M, V; direct=true)
    add_constraint!(constraints, initial_state_constraint, 1:1)
    # dynamics constraint for direct solve
    dynamics_constraint = DynamicsConstraint(IR, model, ts, ix, iu, n, m, M, V)
    add_constraint!(constraints, dynamics_constraint, 2:N-1)
    # create convals from constraints
    convals = convals_from_constraint_list(constraints)
    # put it all together
    Tix = typeof(ix)
    Tiu = typeof(iu)
    T = eltype(X[1])
    TE = typeof(E)
    return Problem{IR,T,Tm,To,Tx,Tix,Tu,Tiu,Tt,TE,TM,TMd,TV}(
        n, m, N, model, obj, convals, X, X_tmp, ix, U, U_tmp, iu, ts, E, M, Md, V
    )
end

"$(TYPEDSIGNATURES)
Get number of states, controls, and knot points"
Base.size(prob::Problem) = size(prob.model)..., prob.N
