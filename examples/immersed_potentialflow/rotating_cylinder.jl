using Taiga, NURBS, StaticArrays, UnivariateSplines, Algoim, SparseArrays
using WriteVTK, IgaBase, ForwardDiff

import Taiga.ImmersedPotentialFlowModule: StreamFunction

# benchmark
r(x, y) = abs(x + im * y)
θ(x, y) = angle(x + im * y)
U, R, ρ, p∞, Γ = 1.0, 1.0, 1.0, 1.0, 6.0*pi;
Ψ = ScalarFunction((x,y) -> -U * r(x,y) * sin(θ(x,y)) * (R^2/r(x,y)^2 - 1) + Γ/(2π) * log(r(x,y)/R))
∇Ψ = Field(
    (x, y) -> ForwardDiff.derivative(x -> Ψ.data(x,y), x),
    (x, y) -> ForwardDiff.derivative(y -> Ψ.data(x,y), y)
)
v = Field((x,y) -> ∇Ψ.data[2](x,y), (x,y) -> -∇Ψ.data[1](x,y))
p = ScalarFunction((x, y) -> p∞ + 0.5*ρ * (U^2 - (v[1].data(x,y)^2 + v[2].data(x,y)^2) ))

F = ImmersedPotentialFlowModule.embedding(8.0, 8.0)
F = refine(F, method=pRefinement(1))
F = refine(F, method=hRefinement(32))

ϕ = AlgoimCallLevelSetFunction(
    x -> -x[1]*x[1] - x[2]*x[2] + R^2,  # ϕ
    x -> [-2.0*x[1], -2.0*x[2]]         # ∇ϕ
)

Dim = dimension(F)
Ω = domain(F)

C = ScalarFunctionSpaceConstraint{Dim}()
left_constraint!(C, 1; dim=1);
right_constraint!(C, 1; dim=1);
left_constraint!(C, 1; dim=2);
right_constraint!(C, 1; dim=2);

S = ScalarSplineSpace(F.space)
Ψʰ = Field(S)
Ψ̄ʰ = Field(S)

t = (x, y) -> begin
    @SVector [ ∇Ψ[1].data(x,y), ∇Ψ[2].data(x,y) ] # traction
end

Ψ̄ = GeometricMapping(Ω, (x,y) -> Ψ(x,y))
for side = 1:2Dim
    project!(boundary(Ψ̄ʰ[1],side), onto=boundary(Ψ̄,side), method=Interpolation)
end
#project!(Ψ̄ʰ, onto=Field(Ψ̄), method=GalerkinProjection)

@info "Model definition..."
model = ImmersedPotentialFlow(F, S, C, Ψʰ, Ψ̄ʰ, t, ϕ);

@info "Linear operator construction..."
@time L = linear_operator(model)

@info "Forcing assembly..."
b = forcing(L, model)

@info "Solution using direct solver..."
@time x = sparse(L) \ b

x = apply_particular_solution(L, model, x)
setcoeffs!(Ψʰ, S, x)

@info "Postprocessing..."
velocity = ImmersedPotentialFlowModule.Velocity{StreamFunction}(F, Ψʰ)
pressure = ImmersedPotentialFlowModule.Pressure(velocity, 1.0; ρ=1.0, p=1.0)

pʰ = Field(S)
vʰ = Field(VectorSplineSpace(S,S))

project!(ϕ, pʰ; onto=pressure, method=GalerkinProjection)
project!(ϕ, vʰ; onto=velocity, method=GalerkinProjection)

@info "Computation of L₂ error norm (u)..."
L₂ = l2_error(ϕ, Ψʰ, to=Ψ, relative=true)[1]
@info "L₂ error norm (u): " L₂

@info "Computation of L₂ error norm (v)..."
L₂ = l2_error(ϕ, vʰ, to=v, relative=true)
@info "L₂ error norm (v): " L₂

@info "Computation of L₂ error norm (v)..."
L₂ = l2_error(ϕ, pʰ, to=p, relative=true)[1]
@info "L₂ error norm (p): " L₂

@info "Bezier extraction..."
vtk_save_bezier("cylinder", F; fields=Dict("Ψʰ" => Ψʰ, "Ψ̄ʰ" => Ψ̄ʰ, "pʰ" => pʰ, "vʰ" => vʰ))

# alternative postprocessing (faster choice, if we don't want to project postprocessed quantities):
# -- vʰ and pʰ are not splines (BezierExtractionContext would interpolate...)
#density = 200
#x = CartesianProduct(i -> IncreasingRange(i..., density), Ω)
#@evaluate streamfunction = Ψʰ(x)
#@evaluate streamfunction_ref = Ψ(x)
#@evaluate pressure = pʰ(x)
#@evaluate velocity = vʰ(x)
#vtk_grid("cylinder", x.data...) do vtk
#    vtk["Stream function"] = streamfunction.data[1]
#    vtk["Pressure"] = pressure.data[1]
#    vtk["Velocity"] = (velocity.data..., zeros(size(x)))
#    vtk["Stream function (ref)"] = streamfunction_ref.data[1]
#end;