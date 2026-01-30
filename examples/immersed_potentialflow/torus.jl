using Taiga, NURBS, StaticArrays, UnivariateSplines, Algoim, SparseArrays
using WriteVTK

F = ImmersedPotentialFlowModule.embedding(12.0, 12.0, 12.0)
F = refine(F, method=pRefinement(2))
F = refine(F, method=hRefinement(16))

# sphere
#ϕ = AlgoimCallLevelSetFunction(
#    x -> -x[1]*x[1] - x[2]*x[2] - x[3]*x[3] + 1.0,  # ϕ
#    x -> [-2.0*x[1], -2.0*x[2], -2.0*x[3]]         # ∇ϕ
#)

# torus
R₁, R₂ = 2.0, 1.0
ϕ = AlgoimCallLevelSetFunction(
    x -> 4 * R₁*R₁ * (x[1]*x[1] + x[2]*x[2]) - (x[1]*x[1]  + x[2]*x[2] + x[3]*x[3] + R₁*R₁ - R₂*R₂) * (x[1]*x[1]  + x[2]*x[2] + x[3]*x[3] + R₁*R₁ - R₂*R₂),  # ϕ
    x -> [
        8 * R₁*R₁ * x[1] - 4*x[1]*(x[1]*x[1]  + x[2]*x[2] + x[3]*x[3] + R₁*R₁ - R₂*R₂),
        8 * R₁*R₁ * x[2] - 4*x[2]*(x[1]*x[1]  + x[2]*x[2] + x[3]*x[3] + R₁*R₁ - R₂*R₂),
        -4*x[3]*(x[1]*x[1]  + x[2]*x[2] + x[3]*x[3] + R₁*R₁ - R₂*R₂)
    ] # ∇ϕ
)

Dim = dimension(F)
Ω = domain(F)

C = ScalarFunctionSpaceConstraint{Dim}()
left_constraint!(C, 1; dim=1);
right_constraint!(C, 1; dim=1);
left_constraint!(C, 1; dim=2);
right_constraint!(C, 1; dim=3);
left_constraint!(C, 1; dim=3);
right_constraint!(C, 1; dim=3);

S = ScalarSplineSpace(F.space)

uʰ = Field(S)
ūʰ = Field(S)

ū = GeometricMapping(Ω, (x,y,z) -> z)
t = (x, y, z) -> @SVector [0.0; 0.0; 0.0]

for side = 1:2Dim
    project!(boundary(ūʰ[1],side), onto=boundary(ū,side), method=Interpolation)
end

@info "Model definition..."
model = ImmersedPotentialFlow(F, S, C, uʰ, ūʰ, t, ϕ);

@info "Linear operator construction..."
@time L = linear_operator(model)

@info "Forcing assembly..."
b = forcing(L, model)

@info "Solution with CG..."
solver = TaigaCG(L, atol=10e-8, rtol=0, itmax=2500, history=true)
@time x, stats = linsolve!(solver, b)
@info stats

x = apply_particular_solution(L, model, x)
setcoeffs!(uʰ, S, x)

@info "Solution Bezier extraction..."
vtk_save_bezier("torus", F; fields=Dict("uʰ" => uʰ, "ūʰ" => ūʰ))

@info "Postprocessing..."
velocity = ImmersedPotentialFlowModule.Velocity(F, uʰ)
pressure = ImmersedPotentialFlowModule.Pressure(velocity, 1.0; ρ=1.0, p=1.0)

#pʰ = Field(S)
#vʰ = Field(VectorSplineSpace(S,S,S))
#
#project!(ϕ, pʰ; onto=pressure, method=GalerkinProjection)
#project!(ϕ, vʰ; onto=velocity, method=GalerkinProjection)
#
#@info "Bezier extraction..."
#vtk_save_bezier("torus", F; fields=Dict("potential" => uʰ, "boundary_condition" => ūʰ, "pressure" => pʰ, "velocity" => vʰ))


# this does not require projection and is easier to handle in paraview:
density = 50
x = CartesianProduct(i -> IncreasingRange(i..., density), Ω)
@evaluate U = uʰ(x)
@evaluate P = pressure(x)
@evaluate V = velocity(x)
vtk_grid("torus", x.data...) do vtk
    vtk["Potential"] = U.data[1]
    vtk["Pressure"] = P.data[1]
    vtk["Velocity"] = (V.data...,)
end; # vʰ and pʰ are not splines (BezierExtractionContext would (badly) interpolate...)