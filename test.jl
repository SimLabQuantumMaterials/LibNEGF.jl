using Pkg

supportedHWs = ["cpu", "apple"]
# unsupportedHWs = ["nvidia", "amd", "intel"]

if ARGS[1] ∉ supportedHWs
    println("The chosen hardware (", ARGS[1], ") is not yet supported")
    exit()
end

Pkg.activate(".");
# passing the hardware type as argument, this creates the
# ARGS variable which is accessible to all of the testsets
Pkg.test(; test_args=[ARGS[1],]);