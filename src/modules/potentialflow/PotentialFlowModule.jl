module PotentialFlowModule
    using Taiga
    using IgaFormation, IgaBase, AbstractMappings, KroneckerProducts, UnivariateSplines, SpecialSpaces
    using SparseArrays, LinearAlgebra

    export PotentialFlow

    include("base.jl")
    include("constitutive.jl")
    include("assemble.jl")
    include("preconditioning.jl")
    include("postprocessing.jl")
end
# Taiga integration
using .PotentialFlowModule
export PotentialFlowModule, PotentialFlow