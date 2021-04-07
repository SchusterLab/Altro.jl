############################################################################################
#                           AUGMENTED LAGRANGIAN OBJECTIVE                                 #
############################################################################################

struct ALObjective{T,O<:Objective} <: AbstractObjective
    obj::O
    constraints::ALConstraintSet{T}
end

function ALObjective(obj::Objective, cons::ConstraintList, model::AbstractModel)
    ALObjective(obj, ALConstraintSet(cons, model))
end
@inline ALObjective(prob::Problem) = ALObjective(prob.obj, prob.constraints, prob.model)

@inline TO.get_J(obj::ALObjective) = obj.obj.J
@inline Base.length(obj::ALObjective) = length(obj.obj)
@inline RobotDynamics.state_dim(obj::ALObjective) = RobotDynamics.state_dim(obj.obj)
@inline RobotDynamics.control_dim(obj::ALObjective) = RobotDynamics.control_dim(obj.obj)


function Base.copy(obj::ALObjective)
    ALObjective(obj.obj, ConstraintSet(copy(obj.constraints.constraints), length(obj.obj)))
end

function TO.cost!(obj::ALObjective, Z::AbstractTrajectory)
    # Calculate unconstrained cost
    TO.cost!(obj.obj, Z)

    # Calculate constrained cost
    TO.evaluate!(obj.constraints, Z)
    TO.update_active_set!(obj.constraints, Val(0.0))
    TO.cost!(TO.get_J(obj), obj.constraints)
end

function TO.cost_expansion!(E::QuadraticObjective, obj::ALObjective, Z::Traj, init::Bool=false, rezero::Bool=false)
    # Update constraint jacobians
    TO.jacobian!(obj.constraints, Z)

    # Calculate expansion of original objective
    TO.cost_expansion!(E, obj.obj, Z, true, rezero)

    # Add in expansion of constraints
    TO.cost_expansion!(E, obj.constraints, Z, true)
end
