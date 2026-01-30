@testset "Doctests" begin
    DocMeta.setdocmeta!(Taiga, :DocTestSetup,
        :(using
            Taiga,
            LinearAlgebra,
            SortedSequences,
            CartesianProducts,
            KroneckerProducts,
            NURBS); recursive=true)
    doctest(Taiga; manual=true)
end