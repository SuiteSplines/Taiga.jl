module LinearElasticity
    using Taiga
    using IgaFormation, IgaBase, AbstractMappings, KroneckerProducts, UnivariateSplines, SpecialSpaces
    using StaticArrays, SparseArrays, LinearAlgebra, LinearMaps
    using ProgressMeter

    include("constitutive.jl")
    include("pullbacks.jl")
    include("postprocessing.jl")
    include("base.jl")
    include("assemble.jl")
    include("preconditioning.jl")
    include("benchmarks.jl")
end
# Taiga integration
using .LinearElasticity
export LinearElasticity