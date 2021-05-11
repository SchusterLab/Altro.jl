"""
constraint_list.jl
"""

############################################################################################
#					             CONSTRAINT LIST										   #
############################################################################################
"""
	AbstractConstraintSet

Stores constraint error and Jacobian values, correctly accounting for the error state if
necessary.

# Interface
- `get_convals(::AbstractConstraintSet)::Vector{<:ConVal}` where the size of the Jacobians
	match the full state dimension
- `get_errvals(::AbstractConstraintSet)::Vector{<:ConVal}` where the size of the Jacobians
	match the error state dimension
- must have field `c_max::Vector{<:AbstractFloat}` of length `length(get_convals(conSet))`

# Methods
Once the previous interface is defined, the following methods are defined
- `Base.iterate`: iterates over `get_convals(conSet)`
- `Base.length`: number of independent constraints
- `evaluate!(conSet, Z::Traj)`: evaluate the constraints over the entire trajectory `Z`
- `jacobian!(conSet, Z::Traj)`: evaluate the constraint Jacobians over the entire trajectory `Z`
- `error_expansion!(conSet, model, G)`: evaluate the Jacobians for the error state using the
	state error Jacobian `G`
- `max_violation(conSet)`: return the maximum constraint violation
- `findmax_violation(conSet)`: return details about the location of the maximum
	constraint violation in the trajectory
"""
abstract type AbstractConstraintSet end

"""
	ConstraintList

Stores the set of constraints included in a trajectory optimization problem. Includes a list
of both the constraint types [`AbstractConstraint`](@ref) as well as the knot points at which
the constraint is applied. Each constraint is assumed to apply to a contiguous set of knot points.

A `ConstraintList` supports iteration and indexing over the `AbstractConstraint`s, and
iteration of both the constraints and the indices of the knot points at which they apply
via `zip(cons::ConstraintList)`.

Constraints are added via the [`add_constraint!`](@ref) method, which verifies that the constraint
dimension is consistent with the state and control dimensions of the problem.

The total number of constraints at each knot point can be queried using the
[`num_constraints`](@ref) method.

The constraint list can also be sorted to separate `StageConstraint`s and `CoupledConstraint`s
via the `sort!` method.

A constraint list can be queried if it has a `DynamicsConstraint` via
`has_dynamics_constraint(::ConstraintList)`.

# Constructor
	ConstraintList(n::Int, m::Int, N::Int)
"""
struct ConstraintList{TM,TV} <: AbstractConstraintSet
    n::Int
    m::Int
    N::Int
	constraints::Vector{AbstractConstraint}
	inds::Vector{AbstractVector} # active knot points for each constraint in constraints
    M::TM
    V::TV
end

function ConstraintList(n::Int, m::Int, N::Int, M::TM, V::TV) where {TM,TV}
    constraints = AbstractConstraint[]
    inds = AbstractVector[]
    return ConstraintList{TM,TV}(n, m, N, constraints, inds, M, V)
end

# methods
Base.length(cons::ConstraintList) = length(cons.constraints)


"""
	add_constraint!(cons::ConstraintList, con::AbstractConstraint, inds::UnitRange, [idx])

Add constraint `cons` to `ConstraintList` `cons` for knot points given by `inds`.

Use `idx` to determine the location of the constraint in the constraint list.
`idx=-1` (default) adds the constraint at the end of the list.

# Example
Here is an example of adding a goal and control limit constraint for a cartpole swing-up.
```julia
# Dimensions of our problem
n,m,N = 4,1,51    # 51 knot points

# Create our list of constraints
cons = ConstraintList(n,m,N)

# Create the goal constraint
xf = [0,π,0,0]
goalcon = GoalConstraint(xf)
add_constraint!(cons, goalcon, N)  # add to the last time step

# Create control limits
ubnd = 3
bnd = BoundConstraint(n,m, u_min=-ubnd, u_max=ubnd, idx=1)  # make it the first constraint
add_constraint!(cons, bnd, 1:N-1)  # add to all but the last time step

# Indexing
cons[1] === bnd                            # (true)
cons[2] === goal                           # (true)
allcons = [con for con in cons]
cons_and_inds = [(con,ind) in zip(cons)]
cons_and_inds[1] == (bnd,1:n-1)            # (true)
```
"""
function add_constraint!(cons::ConstraintList, con::AbstractConstraint, inds::AbstractVector)
    push!(cons.constraints, con)
    push!(cons.inds, inds)
    return nothing
end
