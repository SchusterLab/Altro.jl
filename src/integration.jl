############################################################################################
#                                  EXPLICIT METHODS 								       #
############################################################################################

function integrate(::Type{Euler}, model::AbstractModel, x::StaticVector, u::StaticVector, t, dt)
	xdot = dynamics(model, x, u, t)
	return x + xdot * dt
end

function integrate(::Type{RK2}, model::AbstractModel, x::StaticVector, u::StaticVector, t, dt)
	k1 = dynamics(model, x,        u, t       )*dt
	k2 = dynamics(model, x + k1/2, u, t + dt/2)*dt
	x + k2
end

function integrate(::Type{RK3}, model::AbstractModel, x::StaticVector, u::StaticVector,
		t, dt)
    k1 = dynamics(model, x,             u, t       )*dt;
    k2 = dynamics(model, x + k1/2,      u, t + dt/2)*dt;
    k3 = dynamics(model, x - k1 + 2*k2, u, t + dt  )*dt;
    x + (k1 + 4*k2 + k3)/6
end

function integrate(::Type{RK4}, model::AbstractModel, x::StaticVector, u::StaticVector, t, dt)
	k1 = dynamics(model, x,        u, t       )*dt
	k2 = dynamics(model, x + k1/2, u, t + dt/2)*dt
	k3 = dynamics(model, x + k2/2, u, t + dt/2)*dt
	k4 = dynamics(model, x + k3,   u, t + dt  )*dt
	x + (k1 + 2k2 + 2k3 + k4)/6
end
