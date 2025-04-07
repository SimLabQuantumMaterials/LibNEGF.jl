function loadMatrix(tag, systemName, E, k)
    # directory where the matrices are
    matsDir = "../matrices/" * systemName * "/built_matrices/"

    if tag == "H" || tag == "S"
        # load H or S
        matName = matsDir * tag * "_" * systemName * "_ik" * string(k) * ".mat"
    else
        # load one of the rest
        matName = matsDir * tag * "_" * systemName * "_iE" * string(E) * "_ik" * string(k) * ".mat"
    end

    filex = matopen(matName)
    A = read(filex, tag)
    return A
end

function buildTFromHS(H, S, Sc, energVal)
    # # now, load T itself but from a file and pre-built
    # listMatsToLoad = ["H", "S", "Sc"]
    # # then, in actual desired precision
    # loadedMats, blockSizes = loadMatrices(systemx, E, k,
    #     listMatsToLoad, precx)
    # H = loadedMats[1]
    # S = loadedMats[2]
    # Sc = loadedMats[3]

    # the convert(...) in the following line is to avoid casting
    # to ComplexF64
    T = convert(typeof(H[1, 1]), energVal) * S - H - Sc
    return T
end

function loadEnergies(systemName)
    if systemName != "3x3"
        error("Supporting the 3x3 system only, for now.")
    end
    energVals = CSV.File("../matrices/" * systemName *
                         "/built_matrices/energ_vals.dat") |> Dict
    return energVals
end

function loadMatrices(systemName, E, k, whichMatsToLoad, baseType)
    if systemName != "3x3"
        error("Supporting the 3x3 system only, for now.")
    end

    # TODO : the following array should also be loaded from a file
    if systemName == "3x3"
        blockSizes = [648, 648, 648, 648, 648, 648, 648, 648, 648, 648]
    end

    outMats = Vector{SparseArrays.SparseMatrixCSC{baseType,Int}}()
    for matx in whichMatsToLoad
        # loading matrix in ComplexF64, as they are all stored in F64
        # for now
        loadedMat = loadMatrix(matx, systemName, E, k)
        # then, a casting is done to baseType and pushed to outMats
        push!(outMats, loadedMat)
    end

    return outMats, blockSizes
end