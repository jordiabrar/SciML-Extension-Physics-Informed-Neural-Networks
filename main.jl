using ModelingToolkit, NeuralPDE, Flux, Optimization, OptimizationOptimisers
using GalacticOptim, Plots

# Define PDE: 1D Burgers' equation -> ∂ₜu + u ∂ₓu - ν ∂ₓₓu = 0
@parameters x t
@variables u(..)
ν = 0.01 / π
Dxx = Differential(x)^2
Dt = Differential(t)

# PDE system
eq  = Dt(u(x,t)) + u(x,t)*Differential(x)(u(x,t)) - ν*Dxx(u(x,t)) ~ 0
bcs = [u(0,t) ~ 0, u(1,t) ~ 0, u(x,0) ~ -sin(π*x)]
domains = [x ∈ Interval(0.0,1.0), t ∈ Interval(0.0,1.0)]

pdesys = PDESystem(eq, bcs, domains, [x,t], [u(x,t)])

# Neural network (PINN)
chain = Flux.Chain(
    Dense(2, 50, Flux.σ),
    Dense(50, 50, Flux.σ),
    Dense(50, 1)
)
discretization = PhysicsInformedNN(chain, QuadratureTraining())

# PINN problem
prob = PINNProblem(pdesys, discretization)

# Solver options
tol = 1e-6
opt = OptimizationOptimisers.Adam(0.001)

guess = init_variables(prob)

overall_prob = Optimization.OptimizationFunction(prob, fg = DiffEqSensitivity.TrackerAdjoint())

# Train
res = Optimization.solve(overall_prob, guess, opt; maxiters=1000, callback = Optimization.Options(cb = (x, f)->println("Loss: ", f)))

# Extract solution
sol = discretization(sol = res.minimizer)

# Evaluate on grid
t = 0.5
xs = 0:0.01:1
us = [sol([x_i, t])[1] for x_i in xs]

# Plot
plot(xs, us, xlabel="x", ylabel="u(x,0.5)", title="Burgers PINN at t=0.5")
savefig("pinn_burgers.png")
println("Plot saved as 'pinn_burgers.png'")