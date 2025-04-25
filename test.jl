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
    Pkg.rm("Metal")
end

# passing the hardware type as argument, this creates the
# ARGS variable which is accessible to all of the testsets
Pkg.test(; test_args=[ARGS[1],]);