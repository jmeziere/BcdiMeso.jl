using Documenter
include("DummyDocs.jl")
using .DummyDocs

makedocs(
    sitename="BcdiMeso.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "BCDI"=>"index.md",
        "BcdiMeso"=>"main.md",
        "Usage"=>"use.md"
    ]
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiMeso.jl.git",
)
