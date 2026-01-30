module ImmersedPotentialFlowModule
    using Taiga
    using IgaFormation, IgaBase, AbstractMappings, KroneckerProducts, UnivariateSplines, SpecialSpaces
    using SparseArrays, LinearAlgebra
    using ImmersedSplines, Algoim
    using StaticArrays
    using ProgressMeter
    using NURBS

    export ImmersedPotentialFlow

    include("base.jl")
    include("constitutive.jl")
    include("assemble.jl")
    include("preconditioning.jl")
    include("postprocessing.jl")
end
# Taiga integration
using .ImmersedPotentialFlowModule
export ImmersedPotentialFlowModule, ImmersedPotentialFlow