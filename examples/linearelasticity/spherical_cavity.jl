using Taiga
using NURBS, UnivariateSplines, WriteVTK
using StaticArrays, LinearAlgebra, SparseArrays, SpecialFunctions

# domain
Ω = Interval(1.0,5.0) ⨱ Interval(0.0,pi/2) ⨱ Interval(0.0,pi/2)

# geometric mapping
F = GeometricMapping(Ω, (r,β,θ) -> r*sin(β)*cos(θ), (r,β,θ) -> r*sin(β)*sin(θ), (r,β,θ) -> r*cos(β))
Dim = dimension(F)

# partitioning into elements
Δ = Partition(Ω, ntuple(d -> 6, Dim))

# initialize benchmark
E, ν = 210e3, 0.3
benchmark = LinearElasticity.benchmark_spherical_cavity(; E=E, ν=ν)

# define field for benchmark displacement
uₐ = Field(
    (x,y,z) -> benchmark.displacement(x,y,z)[1],
    (x,y,z) -> benchmark.displacement(x,y,z)[2],
    (x,y,z) -> benchmark.displacement(x,y,z)[3]
)

# define field for benchmark stress
σₐ = Field{Dim,Dim}(
    (x,y,z) -> benchmark.stress(x,y,z)[1,1],
    (x,y,z) -> benchmark.stress(x,y,z)[2,1],
    (x,y,z) -> benchmark.stress(x,y,z)[3,1],
    (x,y,z) -> benchmark.stress(x,y,z)[1,2],
    (x,y,z) -> benchmark.stress(x,y,z)[2,2],
    (x,y,z) -> benchmark.stress(x,y,z)[3,2],
    (x,y,z) -> benchmark.stress(x,y,z)[1,3],
    (x,y,z) -> benchmark.stress(x,y,z)[2,3],
    (x,y,z) -> benchmark.stress(x,y,z)[3,3],
)

# spline discretization
S = VectorSplineSpace(ScalarSplineSpace(ntuple(d -> 6, Dim), Δ))

## define space constraints
C = VectorSplineSpaceConstraints{Dim}()
clamped_constraint!(C[1], :bottom, :front)
clamped_constraint!(C[2], :bottom, :back)
clamped_constraint!(C[3], :top)

# define solution field
uʰ = Field(S)

# define field to satisfy Dirichlet boundary conditions
ūʰ = Field(S)

# define a field or geometric mapping compatible with boundary conditions
ū = Field((x, y, z) -> 0.0, (x, y, z) -> 0.0, (x,y,z) -> 0.0)

# project ū onto solution space
project!(ūʰ, onto=ū, method=Interpolation)

# define traction
t = benchmark.stress

# define material
material = LinearElasticity.Isotropic((x,y,z) -> E, (x,y,z) -> ν)

@info "Model definition..."
model = LinearElasticity.Model(F, S, C, uʰ, ūʰ, t, material)

@info "Model assembly..."
L = linear_operator(model; show_progress=true)
b = forcing(L, model)

@info "Solution using direct solver..."
x₀ = sparse(L) \ b

# apply apply_particular_solution and set field coeffs
x = apply_particular_solution(L, model, x₀)
setcoeffs!(uʰ, S, x)

# postprocess results
ϵ = Taiga.LinearElasticity.Strain(F, uʰ)
σ = Taiga.LinearElasticity.CauchyStress(ϵ, material)
σᵥ = Taiga.LinearElasticity.VonMisesStress(σ)

# l2 errors
@info "Computation of L₂ error norm (u)..."

L₂ = l2_error(uʰ, to=uₐ ∘ F, relative=true)
@info "L₂ error norm (u): " L₂

L₂ = l2_error(σ, to=σₐ ∘ F, relative=true)
@info "L₂ error norm (u): " L₂

# save VTK
density = 25
X = CartesianProduct(i -> IncreasingRange(i..., density), Ω)
@evaluate x = F(X)
@evaluate displacement = uʰ(X)
vtk_grid("spherical_cavity", x.data...) do vtk
    vtk["displacement"] = (displacement.data...,)
end