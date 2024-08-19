using Documenter, BcdiMeso

makedocs(
    sitename="BcdiMeso.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "BcdiMeso"=>"index.md",
        "Usage"=>"use.md"
    ]
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiMeso.jl.git",
)
