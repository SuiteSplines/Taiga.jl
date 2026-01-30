using Test, SafeTestsets, Documenter
using Taiga

tests = [
    "base",
    "assembly",
    "kroneckeraggregate",
    "kroneckerfactory",
    "models",
    "postprocessing",
    "preconditioning",
    "primitives",
    "approximation",
    "linearsolvers",
    "modules/poisson",
    "modules/potentialflow",
    "modules/immersed_potentialflow",
    "modules/linearelasticity",
    #"doctest",
]

mkpath("build/"; mode = 0o755)

@testset "Taiga.jl" begin

    include(joinpath(dirname(@__FILE__), "testutils.jl"))

    for t in tests
        fp = joinpath(dirname(@__FILE__), "$t.jl")
        println("$fp ...")
        include(fp)
    end
end # @testset