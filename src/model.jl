"""
mode.jl
"""

"""
 	AbstractModel

Abstraction of a model of a dynamical system of the form ẋ = f(x,u), where x is the n-dimensional state vector
and u is the m-dimensional control vector.

Any inherited type must define the following interface:
ẋ = dynamics(model, x, u)
n,m = size(model)
"""
abstract type AbstractModel end

"""
	LieGroupModel <: AbstractModel

Abstraction of a dynamical system whose state contains at least one arbitrary rotation.
"""
abstract type LieGroupModel <: AbstractModel end


"""
	RigidBody{R<:Rotation} <: LieGroupModel

Abstraction of a dynamical system with free-body dynamics, with a 12 or 13-dimensional state
vector: `[p; q; v; ω]`
where `p` is the 3D position, `q` is the 3 or 4-dimension attitude representation, `v` is the
3D linear velocity, and `ω` is the 3D angular velocity.

# Interface
Any single-body system can leverage the `RigidBody` type by inheriting from it and defining the
following interface:
```julia
forces(::MyRigidBody, x, u)  # return the forces in the world frame
moments(::MyRigidBody, x, u) # return the moments in the body frame
inertia(::MyRigidBody, x, u) # return the 3x3 inertia matrix
mass(::MyRigidBody, x, u)  # return the mass as a real scalar
```

Instead of defining `forces` and `moments` you can define the higher-level `wrenches` function
	wrenches(model::MyRigidbody, z::AbstractKnotPoint)
	wrenches(model::MyRigidbody, x, u)

# Rotation Parameterization
A `RigidBody` model must specify the rotational representation being used. Any `Rotations.Rotation{3}`
can be used, but we suggest one of the following:
* `UnitQuaternion`
* `MRP`
* `RodriguesParam`
"""
# abstract type RigidBody{R<:Rotation} <: LieGroupModel end

"Integration rule for approximating the continuous integrals for the equations of motion"
abstract type QuadratureRule end

"Specifier for continuous systems (i.e. no integration)"
abstract type Continuous <: QuadratureRule end

"Integration rules of the form x′ = f(x,u), where x′ is the next state"
abstract type Explicit <: QuadratureRule end

"Integration rules of the form x′ = f(x,u,x′,u′), where x′,u′ are the states and controls at the next time step."
abstract type Implicit <: QuadratureRule end

"Fourth-order Runge-Kutta method with zero-order-old on the controls"
abstract type RK4 <: Explicit end

"Second-order Runge-Kutta method with zero-order-old on the controls"
abstract type RK3 <: Explicit end

"Second-order Runge-Kutta method with zero-order-old on the controls (i.e. midpoint)"
abstract type RK2 <: Explicit end

abstract type Euler <: Explicit end

"Third-order Runge-Kutta method with first-order-hold on the controls"
abstract type HermiteSimpson <: Implicit end

"Default quadrature rule"
const DEFAULT_Q = RK3

"""Default size method for model"""
@inline Base.size(model::AbstractModel) = throw("unimplemented")

"""
Continuous time dynamics
"""
@inline dynamics(model::AbstractModel, x::AbstractVector, u::AbstractVector, t::Real, dt::Real) = (
    throw("unimplemented")
)

""" Compute the discretized dynamics of `model` using
explicit integration scheme `Q<:QuadratureRule`.

Methods:
```
x′ = discrete_dynamics(model, model, z)  # uses $(DEFAULT_Q) as the default integration scheme
x′ = discrete_dynamics(Q, model, x, u, t, dt)
x′ = discrete_dynamics(Q, model, z::KnotPoint)
```

The default integration scheme is stored in `TrajectoryOptimization.DEFAULT_Q`
"""
@inline discrete_dynamics(::Type{Q}, model::AbstractModel, x, u, t, dt) where Q =
    integrate(Q, model, x, u, t, dt)

@inline discrete_dynamics!(x_::AbstractVector, ::Type{Q}, model::AbstractModel, x::AbstractVector,
                           u::AbstractVector, t::Real, dt::Real) where {Q} = (
                               x_ .= discrete_dynamics(Q, model, x, u, t, dt)
)

"""
	        discrete_jacobian!(Q, ∇f, model, z::AbstractKnotPoint)

Compute the `n × (n+m)` discrete dynamics Jacobian `∇f` of `model` using explicit
integration scheme `Q<:QuadratureRule`.
"""
function discrete_jacobian!(A::AbstractMatrix, B::AbstractMatrix,
                            IR, model::AbstractModel, x::AbstractVector,
                            u::AbstractVector, t::T, dt::T) where {T}
    fx(x_) = discrete_dynamics(IR, model, x_, u, t, dt)
    ForwardDiff.jacobian!(A, fx, x)
    fu(u_) = discrete_dynamics(IR, model, x, u_, t, dt)
    ForwardDiff.jacobian!(B, fu, u)
    return nothing
end

############################################################################################
#                               STATE DIFFERENTIALS                                        #
############################################################################################

state_diff(model::AbstractModel, x, x0) = x - x0
function state_diff!(δx::AbstractVector, model::AbstractModel, x::AbstractVector,
                     x0::AbstractVector)
    δx .= x
    δx .-= x0
    return nothing
end
@inline state_diff_jacobian(model::AbstractModel, x::SVector{N,T}) where {N,T} = I
@inline state_diff_size(model::AbstractModel) = size(model)[1]
function state_diff_jacobian!(G::AbstractMatrix, model::AbstractModel, x::AbstractVector)
	for i in 1:length(x)
		G[i,i] = 1
	end
end
