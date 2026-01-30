using Taiga
using IgaBase, NURBS, UnivariateSplines, SpecialSpaces
using StaticArrays, LinearAlgebra, Krylov, SpecialFunctions
using KroneckerProducts

# define mapping
F = pinched_cube(;width=1.0, height=1.0, depth=1.0, c=0.001, α=π/(2+1/128) )

# refine mapping (to examplify the isogeometric paradigm)
F = refine(F, method=pRefinement(1))
F = refine(F, method=hRefinement(8))

# define dimension
Dim = dimension(F)

# get domain mapping is defined on
Ω = domain(F)

# define space constraints
C = ScalarSplineSpaceConstraints{Dim}()
# == clamped_constraint!(C, :front, :back)
left_constraint!(C; dim=3);
right_constraint!(C; dim=3);

# define scalar spline space
S = ScalarSplineSpace(F.space)

# define solution field
uʰ = Field(S)

# define field to satisfy Dirichlet boundary conditions
ūʰ = Field(S)

# define Poisson parameters
f = (x, y, z) -> 0.0; # bodyforce (-Δuʰ = f)
ū = (x, y, z) -> 0.0; # function satisfying boundary conditions
κ = (x, y, z) -> SMatrix{3,3,Float64}(I) # conductivity
t = (x, y, z) -> @SVector [0.0; 0.0; 0.0] # traction

# set first plane to 2 and second plane to 3
ūʰ[1].coeffs[:,:,:] .= 0.0
ūʰ[1].coeffs[:,:,1] .= 2.0
ūʰ[1].coeffs[:,:,end] .= 3.0

# initialize model
@info "Model definition..."
model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t);

# construct linear operator
@info "Linear operator construction..."
L = linear_operator(model; matrixfree=true)

# linear operator approximation
@info "Linear operator approximation construction..."
L̃ = linear_operator_approximation(model; method=CanonicalPolyadic, tol=10e-5, rank=10, ntries=10)

# construct rhs vector
@info "Forcing assembly..."
b = forcing(model)

@info "FastDiagonalization construction..."
P₀ = FastDiagonalization(model; method=Wachspress, niter=2)

@info "InnerPCG construction and checks..."
P₁ = InnerPCG(L̃, P₀; η=10e-4, itmax=300, history=true)

# solve
x0 = zeros(size(L, 1))

@info "Solution with Fast Diagonalization"
solver = TaigaPCG(L, P₀; atol=10e-8, rtol=10e-8, itmax=1000, history=true)
@time x, stats = linsolve!(solver, b; x0=x0)
@info stats

@info "Solution with IPCG"
solver = TaigaIPCG(L, P₁; atol=10e-8, rtol=10e-8, itmax=1000, history=true)
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
