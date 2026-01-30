module Taiga

using ArgCheck

using IgaBase
using IgaFormation
using SortedSequences
using CartesianProducts
using UnivariateSplines
using KroneckerProducts
using AbstractMappings
using TensorProductBsplines
using NURBS
using LinearAlgebra
using LinearMaps
using Krylov
using ArnoldiMethod
using SparseArrays
using StaticArrays
using WriteVTK
using PrettyTables
using Algoim
using ImmersedSplines
using SpecialSpaces

import UnivariateSplines.system_matrix
import Base: *
using LinearMaps: LinearMap
import LinearMaps: _unsafe_mul!, zero
using CatViews
using BlockArrays

# HOSVD
using TensorToolbox: hosvd, full, cp_als
using TensorToolbox: ttensor, ktensor

# minimal Taiga solvers BLAS operations
import Krylov: kdotr, kaxpy!, kaxpby!, kscal!

import IgaBase: project!, project_imp!, GalerkinProjection


# framework
include("base.jl")
include("models.jl")
include("kroneckeraggregate.jl")
include("assembly.jl")
include("kroneckerfactory.jl")
include("primitives.jl")
include("postprocessing.jl")
include("linearsolvers.jl")
include("preconditioning.jl")
include("approximation.jl")

# models
include("modules/example/ExampleModule.jl")
include("modules/poisson/PoissonModule.jl")
include("modules/potentialflow/PotentialFlowModule.jl")
include("modules/immersed_potentialflow/ImmersedPotentialFlowModule.jl")
include("modules/linearelasticity/LinearElasticity.jl")


end # module Taiga
