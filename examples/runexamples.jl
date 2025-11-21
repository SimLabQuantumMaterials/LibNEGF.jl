# some general pre-arrangements
begin
    using Pkg
    Pkg.activate(ARGS[5])
    if ARGS[1] == "cpu"
        Pkg.rm("Metal")
    end
    Pkg.instantiate()
end

# choose one of "rgf", "ddrgf", "keldysh"
whichExample = "ddrgf"

if whichExample == "rgf"
    # sequential RGF example
    include("example_rgf.jl")
elseif whichExample == "ddrgf"
    include("example_ddrgf.jl")
elseif whichExample == "keldysh"
end