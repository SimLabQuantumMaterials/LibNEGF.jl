# some general pre-arrangements
begin
    using Pkg
    Pkg.activate(ARGS[5])
    if ARGS[1] == "cpu"
        Pkg.rm("Metal")
    end
    Pkg.instantiate()
end

# sequential RGF example
include("example_rgf.jl")