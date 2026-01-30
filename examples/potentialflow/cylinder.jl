using Taiga, NURBS, StaticArrays, UnivariateSplines

# define mapping
F = hole_in_square_plate()

# refine mapping (to examplify the isogeometric paradigm)
F = refine(F, method=pRefinement(3))
F = refine(F, method=hRefinement(64))

# define dimension
Dim = dimension(F)

# get domain mapping is defined on
Ω = domain(F)

# define space constraints
C = ScalarFunctionSpaceConstraint{Dim}()
left_constraint!(C, 1; dim=1);
right_constraint!(C, 1; dim=2);
right_constraint!(C, 1; dim=1);
#left_constraint!(C, 1; dim=2); # hole

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
U, R, ρ, p∞ = 1.0, 1.0, 1.0, 1.0;
uₐ = ScalarFunction((x, y) -> U*r(x,y) * (1 + R^2 / r(x,y)^2) * cos(θ(x,y)))
pₐ = ScalarFunction((x, y) -> 0.5 * ρ * U^2 * (2 * R^2/r(x,y)^2 * cos(2θ(x,y)) - R^4/r(x,y)^4) + p∞)
vₐ = Field((x,y) -> U + U * R * (y^2 - x^2) / (x^2 + y^2)^2, (x,y) -> -2 * U * R^2 * x * y / (x^2 + y^2)^2 )

# define parameters
t = (x, y) -> @SVector [0.0; 0.0] # traction

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
pʰ = PotentialFlowModule.Pressure(vʰ, U; ρ=ρ, p=p∞)

L₂ = l2_error(uʰ, to=uₐ ∘ F, relative=true)[1]
@info "L₂ error norm (u): " L₂

L₂ = l2_error(pʰ, to=pₐ ∘ F, relative=true)[1]
@info "L₂ error norm (p): " L₂

L₂ = l2_error(vʰ, to=vₐ ∘ F, relative=true)
@info "L₂ error norm (v₁): " L₂[1] L₂[2]

# perform Bezier extraction and save results using convenience functions
@info "Exporting results..."
vtk_save_bezier("results", F; fields=Dict("uʰ" => uʰ, "ūʰ" => ūʰ, "vʰ" => vʰ, "pʰ" => pʰ))
