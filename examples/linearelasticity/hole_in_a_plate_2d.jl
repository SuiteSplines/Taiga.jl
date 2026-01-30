using Taiga
using NURBS, UnivariateSplines, SpecialSpaces, AbstractMappings
using StaticArrays, LinearAlgebra, Krylov, SpecialFunctions

# define mapping
F = hole_in_square_plate()

# refine mapping
F = refine(F, method=pRefinement(1))
F = refine(F, method=hRefinement(8))

# define dimension
Dim = dimension(F)

# get parameteric domain
Ω = domain(F)

# initialize benchmark
E, ν = 210e3, 0.3
benchmark = LinearElasticity.benchmark_hole_in_plate_2d(; E=E, ν=ν)

# define field for benchmark displacement
uₐ = Field((x,y) -> benchmark.displacement(x,y)[1], (x,y) -> benchmark.displacement(x,y)[2])

# define field for benchmark stress
σₐ = Field{Dim,Dim}(
    (x,y) -> benchmark.stress(x,y)[1,1],
    (x,y) -> benchmark.stress(x,y)[2,1],
    (x,y) -> benchmark.stress(x,y)[1,2],
    (x,y) -> benchmark.stress(x,y)[2,2]
)

# define space constraints
#C₁ = ScalarFunctionSpaceConstraint{Dim}()
#C₂ = ScalarFunctionSpaceConstraint{Dim}()
#left_constraint!(C₁, 1; dim=1);
#right_constraint!(C₂, 1; dim=1);
#C = VectorFunctionSpaceConstraint(C₁, C₂)
C = VectorSplineSpaceConstraints{Dim}()
clamped_constraint!(C[1], :left)
clamped_constraint!(C[2], :right)


# define vector spline space
S = VectorSplineSpace(ScalarSplineSpace(F.space))

# define solution field
uʰ = Field(S)

# define field to satisfy Dirichlet boundary conditions
ūʰ = Field(S)

# define a field or geometric mapping compatible with boundary conditions
ū = Field((x, y) -> 0.0, (x, y) -> 0.0)

# project ū onto solution space
project!(ūʰ, onto=ū, method=Interpolation)

# define traction
t = benchmark.stress

# define material
material = LinearElasticity.PlaneStrain((x,y) -> E, (x,y) -> ν)

@info "Model definition..."
model = LinearElasticity.Model(F, S, C, uʰ, ūʰ, t, material)

@info "Model assembly..."
L = linear_operator(model; show_progress=true)
b = forcing(L, model)

@info "Preconditioner construction..."
P = FastDiagonalization(model)

@info "Solving linear system of equations using TaigaPCG..."
solver = TaigaPCG(L, P; itmax=1000, atol=1.0e-8, rtol=0.0)
@time x₀, stats = linsolve!(solver, b)
@info stats

# define solution field
uʰ = Field(S)

# apply apply_particular_solution and set field coeffs
x = apply_particular_solution(L, model, x₀)
setcoeffs!(uʰ, S, x)

# postprocess results
ε = Taiga.LinearElasticity.Strain(F, uʰ)
σ = Taiga.LinearElasticity.CauchyStress(ε, material)
σᵥ = Taiga.LinearElasticity.VonMisesStress(σ)

# l2 errors
@info "Computation of L₂ error norm (u)..."

L₂ = l2_error(uʰ, to=uₐ ∘ F, relative=true)
@info "L₂ error norm (u): " L₂

L₂ = l2_error(σ, to=σₐ ∘ F, relative=true)
@info "L₂ error norm (u): " L₂

# export vtk
vtk_save_bezier("results", F; fields = Dict("uʰ" => uʰ, "ūʰ" => ūʰ, "σ" => σ, "σᵥ" => σᵥ))
