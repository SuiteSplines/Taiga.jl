using Taiga
using NURBS, UnivariateSplines
using StaticArrays, LinearAlgebra, Krylov, SpecialFunctions
using KroneckerProducts

# define mapping
F = curved_rectangle()

# refine mapping (to examplify the isogeometric paradigm)
F = refine(F, method=pRefinement(2))
F = refine(F, method=hRefinement(32))

# define dimension
Dim = dimension(F)

# get domain mapping is defined on
Ω = domain(F)

# define space constraints
C = ScalarSplineSpaceConstraints{Dim}()
clamped_constraint!(C, :left, :right, :top, :bottom)

# define scalar spline space
S = ScalarSplineSpace(F.space)

# define solution field
uʰ = Field(S)

# define field to satisfy Dirichlet boundary conditions
ūʰ = Field(S)

# define Poisson parameters
f = (x, y) -> 1.0; # bodyforce (-Δuʰ = f)
ū = (x, y) -> 0.0; # function satisfying boundary conditions
κ = (x, y) -> SMatrix{2,2,Float64}(begin
    θ = pi / 4 # principal diffusion axis rotation
    R = [cos(θ) -sin(θ); sin(θ) cos(θ)] # rotation matrix
    U = R * [1 0; 0 1] # basis vectors for diffusion
    Λ = Diagonal([5.0, 0.001]) # diffusion magnitudes in principal directions
    κ = U * Λ * U'
end) # conductivity
t = (x, y) -> @SVector [0.0; 0.0] # traction

# project ū onto solution space
project!(ūʰ, onto=Field(ū), method=Interpolation)

# initialize model
@info "Model definition..."
model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t);

# construct linear operator
@info "Linear operator construction..."
L = linear_operator(model; matrixfree=true)

# linear operator approximation
@info "Linear operator approximation construction..."
L̃ = linear_operator_approximation(model; method=CanonicalPolyadic, tol=10e-5, rank=4)

# construct rhs vector
@info "Forcing assembly..."
b = forcing(model)

@info "FastDiagonalization construction..."
P₀ = FastDiagonalization(model; method=Wachspress, niter=2)

@info "InnerPCG construction and checks..."
P₁ = InnerPCG(L̃, P₀; η=10e-2, itmax=300, history=true)

# solve
x0 = zeros(size(L, 1))

@info "Solution with Fast Diagonalization"
solver = TaigaPCG(L, P₀; atol=10e-8, rtol=10e-8, itmax=300, history=true)
@time x, stats = linsolve!(solver, b; x0=x0)
@info stats

@info "Solution with IPCG"
solver = TaigaIPCG(L, P₁; atol=10e-8, rtol=10e-8, itmax=300, history=true)
reset_inner_solver_history!(P₁)
@time x, stats = linsolve!(solver, b; x0=x0)
@info stats
@info "InnerPCG required $(join(inner_solver_niters(P₁), ", ", " and ")) iterations." 

# apply particular solution
x = apply_particular_solution(model, x)

# apply solution to field coefficients
setcoeffs!(uʰ, S, x)

# perform Bezier extraction and save results using convenience functions
@info "Exporting results..."
vtk_save_bezier("results", F; fields=Dict("uʰ" => uʰ))
