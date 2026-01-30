```@setup using
using Taiga, LinearAlgebra
using SortedSequences, CartesianProducts, KroneckerProducts, NURBS, SpecialSpaces
```
```@meta
DocTestSetup = quote
    using Taiga
    using LinearAlgebra
    using SortedSequences, CartesianProducts, KroneckerProducts, NURBS
end
```

# Taiga.jl

Tensor-product applications in isogeometric analysis.

!!! note
    Examples listed on this page use following packages
    ```julia
    using Taiga, LinearAlgebra
    using SortedSequences, CartesianProducts, KroneckerProducts, NURBS, SpecialSpaces
    ```


## Postprocessing
Some basic postprocessing methods are provided as `Plots` recipes in most
packages in the `Feather` ecosystem. `Taiga.jl` implements Bézier extraction
and can save mappings, fields and control nets to a `VTK` file, e.g

```jldoctest; output=false
mapping = hole_in_square_plate()
mapping = refine(mapping, method=hRefinement(3))
space = ScalarSplineSpace(mapping.space)

euclidean_distance = Field((x,y) -> sqrt(x^2 + y^2))
d = Field(space)
project!(d, onto=euclidean_distance ∘ mapping; method=QuasiInterpolation)

vtk_save_bezier("build/geometry_and_field", mapping; fields = Dict("d" => d))
vtk_save_control_net("build/control_net", mapping)

# output
1-element Vector{String}:
 "build/control_net.vtu"
```

![Bezier extraction on square plate with a hole](assets/square_plate_with_hole_bezier_example.png)

More detailed examples are provided in the `examples` directory.

## Aggregation of Kronecker products

More often then not Kronecker product approximations will be composed of a sum
of Kronecker products. In most cases these sums will not have the classical
[Kronecker sum](https://en.wikipedia.org/wiki/Kronecker_product) structure.
Nonetheless, it is still possible to evaluate matrix-vector products with
these operators with complexity ``\mathcal O(M\cdot N^{3/2})`` and 
``\mathcal O(M\cdot N^{4/3})`` in 2 and 3 dimensions, where ``M`` is the number
of Kronecker products in the collection.

[`KroneckerProductAggregate`](@ref) is a collection of Kronecker product
matrices of conforming size that acts as a linear map and supports reduction
over the collection, e.g.

```jldoctest
julia> A = rand(7, 3) ⊗ rand(5, 3) ⊗ rand(3, 5);

julia> B = rand(7, 3) ⊗ rand(5, 3) ⊗ rand(3, 5);

julia> C = A + B;

julia> K = KroneckerProductAggregate(A, B);

julia> v = rand(size(C, 2));

julia> b = K * v;

julia> b ≈ C * v
true

julia> mul!(b, K, v); # in-place evaluation

julia> b ≈ C * v
true
```

A [`KroneckerProductAggregate`](@ref) supports `LinearAlgebra.mul!`,
`LinearAlgebra.adjoint`, `Base.isempty`, `Base.push!` and most other 
methods that abstract matrices implement.

## `KroneckerFactory`

Let us consider that we need to assemble the bilinear form associated with the operator
`L(u) = ∇ ⋅ (C∇u)` where `C = [6.0 0.1875; 0.125 12.0]` is a conductivity matrix.
Clearly, even on Cartesian grids an exact [`FastDiagonalization`](@ref) of the resulting
matrix is not possible due to non-zero off-diagonal entries in `C`. Furthermore, if `C`
is a function of the spatial coordinates, the resulting stiffness matrix does not have
a Kronecker product structure at all and the material pullback needs to be approximated
using some separable approximation.

[`KroneckerFactory`](@ref) can be used to obtain approximations of a bilinear form given
a separable approximation of the data, i.e. weighted spline approximation of the pullback
operator on mapped geometries or Wachspress approximation at quadrature points.

The syntax is close to that of `IgaFormation.Sumfactory`.

```julia
∇u = k -> ι(k, dim=Dim)
∇v = k -> ι(k, dim=Dim)

∫ = KroneckerFactory(S, S)
for β in 1:Dim
    for α in 1:Dim
        ∫(∇u(α), ∇v(β); data=f[α, β])
    end
end

K = ∫.data
```

In the above, `f[α, β]` is a `Tuple` of either univariate weighting splines in each
parametric direction or plain vectors with weighting coefficients. This data is used
to weight the test functions. 

Similarly to `Sumfactory`, the data collection can be reset
```julia
∫(∇u(α), ∇v(β); data=f[α, β], reset=true)
```

or simply

```julia
reset!(∫)
```

## Linear solvers

Taiga provides minimal reference implementations of linear solvers:

- [`TaigaCG`](@ref) Conjugate Gradient method
- [`TaigaPCG`](@ref) preconditioned Conjugate Gradient method
- [`TaigaIPCG`](@ref) inexactly preconditioned Conjugate Gradient method

These implementations are reasonably fast. The solution times are at least as good as `Krylov.jl`,
whereby due to the minimal implementation there are less memory allocations. Furthermore, some
solvers like [`TaigaIPCG`](@ref) are nowhere to be found and are essential to Taiga's conceptual framework.

```julia-repl
julia> solver = TaigaPCG(L, P; atol=10e-8, rtol=10e-8, itmax=100)
Linear solver of type TaigaPCG (atol=1.0e-7, rtol=1.0e-7, itmax=100)

julia> x, stats = linsolve!(solver, b);

julia> stats
Linear solver statistics for TaigaPCG:
┌──────────────────┬───────────────────────────────┐
│ Metric           │ Value                         │
├──────────────────┼───────────────────────────────┤
│ converged        │ true                          │
│ niter            │ 47                            │
│ residual_norm    │ 9.25797e-8                    │
│ residual_norm_x₀ │ 0.0732086                     │
│ status           │ coverged: atol ✔, rtol ✘      │
└──────────────────┴───────────────────────────────┘
```

## Index

```@autodocs
Modules = [Taiga]
Order   = [:type, :function]
```

```@meta
DocTestSetup = nothing
```