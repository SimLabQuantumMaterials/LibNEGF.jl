using Metal

"""
	ArrayOrLU_ = Union{Array, LU, UndefInitializer}

Type for a matrix that can contain an `Array`, `LU factor` and/or `undef`.
"""
ArrayOrLU_ = Union{Array,LU,UndefInitializer}

"""
	ArrayOrLUDev_ = Union{Array, LU, UndefInitializer}

Type for a matrix that can contain an `Array`, `LU factor` and/or `undef`,
with blocks stored in the (GPU) device.
"""
ArrayOrLUDev_ = Union{Metal.MtlArray,LU,UndefInitializer}

# TODO : documentation
function be_copy_to_hw(M::Array)::Metal.MtlArray
    return Metal.MtlArray(M)
end

# TODO : documentation
function be_copy_from_hw(M::Metal.MtlArray)::Array
    return Array(M)
end

# TODO : documentation
function be_inv(M::Metal.MtlArray)::Metal.MtlArray
    # for now, we have to emulate this i.e. copy to CPU, do
    # things on the CPU, and copy back to Metal
    Mcpu = Array(M)
    McpuInv = inv(Mcpu)
    Mout = Metal.MtlArray(McpuInv)
    return Mout
end

# TODO : documentation
function be_copy_to_hw!(Mout::Metal.MtlArray, Min::Array)
    copy!(Mout, Min)
end

# TODO : documentation
function be_copy_from_hw!(Mout::Array, Min::Metal.MtlArray)
    copy!(Mout, Min)
end

# TODO : documentation
function be_copy_in_hw!(Mout::Metal.MtlArray, Min::Metal.MtlArray)
    copy!(Mout, Min)
end