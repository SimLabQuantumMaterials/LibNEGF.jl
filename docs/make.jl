# run this script from within docs/ as:
# julia --color=yes --project make.jl

using Pkg

supportedHWs = ["cpu", "apple"]
# unsupportedHWs = ["nvidia", "amd", "intel"]

if ARGS[1] ∉ supportedHWs
    println("The chosen hardware (", ARGS[1], ") is not yet supported")
    exit()
end

Pkg.activate("./")
Pkg.develop(path="../")

using Documenter, LibNEGF, SparseArrays

makedocs(sitename="LibNEGF", remotes=nothing)