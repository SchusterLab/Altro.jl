"""
Altro.jl
"""

module Altro

using BenchmarkTools
using Crayons
using DocStringExtensions
using ForwardDiff
using Interpolations
using LinearAlgebra
using Logging
using Parameters
using RecipesBase
using SparseArrays
using SolverLogging
using StaticArrays
using Statistics
using UnsafeArrays

include("utils.jl")
include("model.jl")
include("integration.jl")
include("costfunctions.jl")
include("objective.jl")
include("constraints.jl")
include("constraint_list.jl")
include("convals.jl")
include("problem.jl")
include("solvers.jl")
include("solver_opts.jl")
include("ilqr/ilqr.jl")
include("ilqr/backwardpass.jl")
include("ilqr/rollout.jl")
include("ilqr/ilqr_methods.jl")
include("al/al.jl")
include("al/al_methods.jl")
include("al/al_objective.jl")
include("pn/pn.jl")
include("pn/pn_methods.jl")
include("altro/altro.jl")

export
    # solver
    ALTROSolver,
    SolverStats, SolverOptions,
    solve!, benchmark_solve!, states, controls,
    # constraints
    ConstraintList, add_constraint!,
    GoalConstraint, BoundConstraint,
    # problem
    Problem,
    # model
    AbstractModel,
    Explicit, RK4, RK3, RK2, Euler,
    dynamics, discrete_dynamics, discrete_dynamics!, discrete_jacobian!,
    # cost functions
    Objective, LQRObjective
end # module
