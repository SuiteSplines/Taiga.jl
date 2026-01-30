using Taiga, NURBS, StaticArrays, UnivariateSplines, Algoim, SparseArrays
using WriteVTK, IgaBase, AbstractMappings, SpecialSpaces


# benchmark
r(x, y) = abs(x + im * y)
θ(x, y) = angle(x + im * y)
U, R, ρ, p∞ = 1.0, 1.0, 1.0, 1.0
uₐ = ScalarFunction((x, y) -> U*r(x,y) * (1 + R^2 / r(x,y)^2) * cos(θ(x,y)))
pₐ = ScalarFunction((x, y) -> 0.5 * ρ * U^2 * (2 * R^2/r(x,y)^2 * cos(2θ(x,y)) - R^4/r(x,y)^4) + p∞)
vₐ = Field((x,y) -> U + U * R * (y^2 - x^2) / (x^2 + y^2)^2, (x,y) -> -2 * U * R^2 * x * y / (x^2 + y^2)^2 )

F = ImmersedPotentialFlowModule.embedding(8.0, 8.0)
F = refine(F, method=pRefinement(1))
F = refine(F, method=hRefinement(256))

ϕ = AlgoimCallLevelSetFunction(
    x -> -x[1]*x[1] - x[2]*x[2] + R^2,  # ϕ
    x -> [-2.0*x[1], -2.0*x[2]]         # ∇ϕ
)

Dim = dimension(F)
Ω = domain(F)

C = ScalarSplineSpaceConstraints{Dim}()
left_constraint!(C; dim=1);
right_constraint!(C; dim=1);
left_constraint!(C; dim=2);
right_constraint!(C; dim=2);

S = ScalarSplineSpace(F.space)

uʰ = Field(S)
ūʰ = Field(S)

ū = GeometricMapping(Ω, (x,y) -> uₐ(x,y))
t = (x, y) -> @SVector [0.0; 0.0] # traction

for side = 1:2Dim
    project!(boundary(ūʰ[1],side), onto=boundary(ū,side), method=Interpolation)
end

@info "Model definition..."
model = ImmersedPotentialFlow(F, S, C, uʰ, ūʰ, t, ϕ);

@info "Linear operator construction..."
@time L = linear_operator(model)

@info "Forcing assembly..."
b = forcing(L, model)

@info "Solution using direct solver..."
@time x = sparse(L) \ b

x = apply_particular_solution(L, model, x)
setcoeffs!(uʰ, S, x)

@info "Postprocessing..."
velocity = ImmersedPotentialFlowModule.Velocity(F, uʰ)
pressure = ImmersedPotentialFlowModule.Pressure(velocity, 1.0; ρ=1.0, p=1.0)

pʰ = Field(S)
vʰ = Field(VectorSplineSpace(S,S))

#project!(ϕ, pʰ; onto=pressure, method=GalerkinProjection)
#project!(ϕ, vʰ; onto=velocity, method=GalerkinProjection)
#
#@info "Computation of L₂ error norm (u)..."
#L₂ = l2_error(ϕ, uʰ, to=uₐ, relative=true)[1]
#@info "L₂ error norm (u): " L₂
#
#L₂ = l2_error(ϕ, pʰ, to=pₐ, relative=true)[1]
#@info "L₂ error norm (p): " L₂
#
#L₂ = l2_error(ϕ, vʰ, to=vₐ, relative=true)
#@info "L₂ error norm (v): " L₂
#
#@info "Bezier extraction..."
#vtk_save_bezier("cylinder", F; fields=Dict("potential" => uʰ, "boundary_conditions" => ūʰ, "pressure" => pʰ, "velocity" => vʰ))

# alternative postprocessing (faster choice, if we don't want to project postprocessed quantities):
# -- vʰ and pʰ are not splines (BezierExtractionContext would interpolate...)
density = 300
x = CartesianProduct(i -> IncreasingRange(i..., density), Ω)
x = Partition(S)
@evaluate potential = uʰ(x)
@evaluate pressure = pʰ(x)
@evaluate velocity = vʰ(x)
vtk_grid("cylinder", x.data...) do vtk
    vtk["Levelset"] = ϕ.(x)
    vtk["Potential"] = potential.data[1]
    vtk["Pressure"] = pressure.data[1]
    vtk["Velocity"] = (velocity.data..., zeros(size(x)))
end;