using Documenter
using Taiga

DocMeta.setdocmeta!(Taiga, :DocTestSetup, :(using Taiga); recursive=true)

makedocs(;
    modules=[Taiga],
    authors="Michał Mika and contributors",
    sitename="Taiga.jl",
    format=Documenter.HTML(;
        canonical="https://SuiteSplines.github.io/Taiga.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Poisson equation" => "poisson.md",
        "Potential flow" => "potentialflow.md",
        "Immersed potential flow" => "immersed_potentialflow.md",
        "Linear elasticity" => "linearelasticity.md",
    ],
    doctest=false,
)
