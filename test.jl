using Pkg

supportedHWs = ["cpu", "apple"]
# unsupportedHWs = ["nvidia", "amd", "intel"]

if ARGS[1] ∉ supportedHWs
    println("The chosen hardware (", ARGS[1], ") is not yet supported")
    exit()
end

Pkg.activate(".");

# depending on the type of hardware, remove packages as necessary
if ARGS[1] != "apple"
    try
        Pkg.rm("Metal")
    catch
        println("Tried to remove Metal but already not in Project.toml")
    end
end

# Passing the hardware type as argument, this creates the
# ARGS variable which is accessible to all of the testsets
# The following call triggers the precompilation of LibNEGF
Pkg.test(; test_args=[ARGS[1], ARGS[2], ARGS[3], ARGS[4], ARGS[5],]);