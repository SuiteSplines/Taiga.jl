using Taiga
using NURBS, UnivariateSplines
using StaticArrays, LinearAlgebra, Krylov, SpecialFunctions

# define mapping
width, height = 2π, exp(1)
F = rectangle(; width=width, height=height)

# refine mapping (to examplify the isogeometric paradigm)
F = refine(F, method=pRefinement(1))
F = refine(F, method=hRefinement(2))

# define dimension
Dim = dimension(F)

# get domain mapping is defined on
Ω = domain(F)

# define space constraints
C = ScalarSplineSpaceConstraints{Dim}()

# define scalar spline space
S = ScalarSplineSpace(F.space)

# define solution field
uʰ = Field(S)

# define field to satisfy Dirichlet boundary conditions
ūʰ = Field(S)

# define analytical solution
R = θ -> [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
e₁ = @SVector [1.0; 0.0]
eᵣ = R(π/6) * e₁ # constant traction vector
l = dot(eᵣ, [width; height])
uₐ = ScalarFunction((x, y) -> dot(eᵣ, [x, y]) - 0.5l)

# define Poisson parameters
f = (x, y) -> 0.0; # bodyforce (-Δuʰ = f)
ū = Field((x, y) -> 0.0); # field satisfying boundary conditions
κ = (x, y) -> SMatrix{2,2,Float64}(I) # conductivity
t = (x, y) -> eᵣ # traction 

# project ū onto solution space
project!(ūʰ, onto=ū, method=Interpolation)

# initialize model
@info "Model assembly..."
model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t);

# construct linear operator
L = linear_operator(model)

# construct rhs vector
b = forcing(L)

# solve unpreconditioned system
@info "Solving linear system of equations..."
x₀, log = cg(L, b)
@info "Solver statistics:" log

# apply particular solution
x = apply_particular_solution(L, x₀)

# apply solution to field coefficients
setcoeffs!(uʰ, S, x)

# compute L₂ error norm
L₂ = l2_error(uʰ, to=uₐ ∘ F; relative=true)[1]
@info "L₂ error norm: " L₂

# perform Bezier extraction and save results using convenience functions
@info "Exporting results..."

uₐʰ = Field(S)
project!(uₐʰ, onto=Field(uₐ ∘ F), method=Interpolation)
vtk_save_bezier("results", F; fields=Dict("uʰ" => uʰ, "uₐʰ" => uₐʰ, "ūʰ" => ūʰ))
