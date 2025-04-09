# run this script from within docs/ as:
# julia --color=yes --project make.jl

using Pkg

Pkg.activate("./")
Pkg.develop(path="../")

using Documenter, LibNEGF, SparseArrays

makedocs(sitename="LibNEGF", remotes=nothing)