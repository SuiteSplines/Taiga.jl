module PoissonModule
    using Taiga
    using IgaFormation, IgaBase, AbstractMappings, KroneckerProducts, UnivariateSplines, SpecialSpaces
    using SparseArrays, LinearAlgebra

    export Poisson

    include("base.jl")
    include("assemble.jl")
    include("constitutive.jl")
    include("preconditioning.jl")
end
# Taiga integration
using .PoissonModule
export PoissonModule, Poisson