using Taiga, NURBS, StaticArrays, UnivariateSplines

# define mapping
F = annulus()

# refine mapping (to examplify the isogeometric paradigm)
F = refine(F, method=pRefinement(1))
F = refine(F, method=hRefinement(32))

# define dimension
Dim = dimension(F)

# get domain mapping is defined on
Ω = domain(F)

# define space constraints
C = ScalarFunctionSpaceConstraint{Dim}()
left_constraint!(C, 1; dim=1);
#right_constraint!(C, 1; dim=1);

# define scalar spline space
S = ScalarSplineSpace(F.space)

# define solution field
uʰ = Field(S)

# define field to satisfy Dirichlet boundary conditions
ūʰ = Field(S)

# define transformation from Cartesian to polar coordinates
r(x, y) = abs(x + im * y)
θ(x, y) = angle(x + im * y)

# define analytical solution
uₐ = ScalarFunction((x, y) -> 2 * (1 + θ(x,y) / π))

# define parameters
t = (x, y) -> begin
    if θ(x,y) >= π/2
        @SVector [-2 * y / (π * (x^2 + y^2)); 2 * x / (π * (x^2 + y^2))]
    else
        @SVector [0; 0]
    end
end
#t = (x,y) -> @SVector [0; 0]

# project ū onto solution space
project!(ūʰ, onto=Field(uₐ ∘ F), method=Interpolation)

# initialize model
@info "Model definition..."
model = PotentialFlow(F, S, C, uʰ, ūʰ, t);

# construct linear operator
@info "Linear operator construction..."
L = linear_operator(model)

# construct rhs vector
@info "Forcing assembly..."
b = forcing(L, model)

@info "FastDiagonalization construction..."
P = FastDiagonalization(model)

# solve
@info "Solution with PCG..."
solver = TaigaPCG(L, P; atol=10e-12, rtol=0, itmax=100, history=true)
@time x, stats = linsolve!(solver, b)
@info stats

# apply particular solution
x = apply_particular_solution(L, model, x)

# apply solution to field coefficients
setcoeffs!(uʰ, S, x)

# postprocessing
vʰ = PotentialFlowModule.Velocity(F, uʰ)

L₂ = l2_error(uʰ, to=uₐ ∘ F, relative=true)[1]
@info "L₂ error norm: " L₂

# perform Bezier extraction and save results using convenience functions
@info "Exporting results..."
vtk_save_bezier("results", F; fields=Dict("uʰ" => uʰ, "ūʰ" => ūʰ, "vʰ" => vʰ))
