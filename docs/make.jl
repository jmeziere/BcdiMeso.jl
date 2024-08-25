using Documenter, DocumenterCitations, BcdiMeso

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename="BcdiMeso.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "BcdiMeso"=>"index.md",
        "Usage"=>"use.md",
        "References"=>"refs.md"
    ],
    plugins = [bib]
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiMeso.jl.git",
)
