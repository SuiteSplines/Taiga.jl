using Documenter
using Taiga

ENV["JULIA_DEBUG"] = "Documenter"

package_info = Dict(
    "modules" => [Taiga],
    "authors" => "Michal Mika",
    "name" => "Taiga.jl",
    "repo" => "https://gitlab.com/m1ka05/taiga.jl",
    "pages" => [
        "Taiga.jl"  =>  "index.md"
        "Poisson equation"  =>  "poisson.md"
        "Potential flow"  =>  "potentialflow.md"
        "Immersed potential flow"  =>  "immersed_potentialflow.md"
        "Linear elasticity"  =>  "linearelasticity.md"
    ],
)

DocMeta.setdocmeta!(Taiga, :DocTestSetup,
    :(using
        Taiga,
        LinearAlgebra,
        SortedSequences,
        CartesianProducts,
        KroneckerProducts,
        NURBS); recursive=true)

if haskey(ENV, "DOC_TEST_DEPLOY") && ENV["DOC_TEST_DEPLOY"] == "yes"
    doctest(Taiga, fix=true)
end

makedocs(
    modules  = package_info["modules"],
    sitename = package_info["name"],
    authors  = package_info["authors"],
    pages    = package_info["pages"],
    repo     = package_info["repo"] * "/blob/{commit}{path}#{line}",
    doctest = false
)
