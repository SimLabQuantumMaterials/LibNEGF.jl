"""
    load_matrix(tag::String, systemName::String, E::Integer, k::Integer)

Load a matrix from the data in `../matrices/`.

The matrices are assumed to be in `../matrices/``, in the sub-directory
specified via the input variable `systemName`, and furthermore the input
variables `tag`, `E` and `k` indicate which matrix to load from within `built_matrices/`.
It returns a sparse matrix.
"""
function load_matrix(tag::String, systemName::String, E::Integer,
    k::Integer)::SparseArrays.SparseMatrixCSC
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
    A::SparseArrays.SparseMatrixCSC = read(filex, tag)
    return A
end

"""
    load_energies(systemName::String)

For the input system in `systemName`, load the energy indices from a file
within `matrices/`.
"""
function load_energies(systemName::String)::Dict{Int,Float64}
    if systemName != "3x3"
        error("Supporting the 3x3 system only, for now.")
    end
    energVals::Dict{Int,Float64} = CSV.File("../matrices/" * systemName *
                                            "/built_matrices/energ_vals.dat") |> Dict
    return energVals
end

"""
    load_matrices(systemName::String, E::Int, k::Int,
       whichMatsToLoad::Vector{String}, baseType::DataType)

For the system in `systemName` and the energy indices in `E` and `k`, load
the matrices listed in `whichMatsToLoad`. The matrix is loaded in ComplexF64
and cast to the desired precision indicated via `baseType`.
"""
function load_matrices(systemName::String, E::Int, k::Int,
    whichMatsToLoad::Vector{String}, baseType::DataType)
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
        loadedMat = load_matrix(matx, systemName, E, k)
        # then, a casting is done to baseType and pushed to outMats
        push!(outMats, loadedMat)
    end

    return outMats, blockSizes
end

"""
    build_T_from_HS(H::SparseArrays.SparseMatrixCSC, S::SparseArrays.SparseMatrixCSC,
        Se::SparseArrays.SparseMatrixCSC, energVal::Float64)

Construct ``T = ES - H - S_e``.
"""
function build_T_from_HS(H::SparseArrays.SparseMatrixCSC, S::SparseArrays.SparseMatrixCSC,
    Se::SparseArrays.SparseMatrixCSC, energVal::Float64)::SparseArrays.SparseMatrixCSC
    # TODO : add a check that the types of S, H and Se are all the same
    # the convert(...) in the following line is to avoid casting
    # to ComplexF64
    T = convert(typeof(H[1, 1]), energVal) * S - H - Se
    return T
end