import TimerOutputs
using TimerOutputs

export TimingData
export CountingData

"""
	TimingData

When timing backend kernels within the most computationally demanding
inversions, this data allows us to enable or disable those finer-level
timings.
"""
struct TimingData
    to::TimerOutput
    label::String
end

# counters for floating point operations and memory accesses
mutable struct CountingData
    flops::Int
    mems::Int
    nrCalls::Int
end

@ifdef "LIBNEGF_HW" begin
    if ENV["LIBNEGF_HW"] == "cpu"
        function flops_and_mems(whichKernel::String, cd::CountingData, sizes)
            if whichKernel == "_gemm"
                m = sizes[3][1]
                n = sizes[3][2]
                k = sizes[1][2]
                cd.flops += 2*m*n*k
                cd.mems += k*(m+n) + 2*m*n
            elseif whichKernel == "_mldivide"
                cd.flops += 0
                cd.mems += 0
            elseif whichKernel == "_mrdivide"
                cd.flops += 0
                cd.mems += 0
            elseif whichKernel == "_lu"
                cd.flops += 0
                cd.mems += 0
            end
        end
    elseif ENV["LIBNEGF_HW"] == "apple"
        # nothing yet
    end
end

"""
	timewrap(tdx, suffx, codex)

This macro allows us to enable or disable finer-level timings, where
the backend kernel calls are wrapped with it.
"""
macro timewrap(tdx, cdx, suffx, sizesx, codex)
    return quote
        flops_and_mems($(esc(suffx)), $(esc(cdx)), $(esc(sizesx))); @timeit ($(esc(tdx))).to ($(esc(tdx))).label * ($(esc(suffx))) $(esc(codex))
    end
end

function print_flops_and_mems(cd::CountingData, to::TimerOutput, prec::DataType)
    nrCalls = cd.nrCalls
    flopsAvg = cd.flops/nrCalls
    memsAvg = cd.mems/nrCalls/1024/1024
    totTimeAvg = ( TimerOutputs.time(to["bndiag_of_inv_direct_"*string(prec)]) * 1.0E-9 ) / nrCalls

    println("\nFlops and mems (" * string(prec) * "):")
    println("\t -- nr calls : " * string(nrCalls))
    println("\t -- flops (avg) : " * string(flopsAvg))
    println("\t -- mems (avg) (MB) : " * string(memsAvg))
    println("\t -- time (avg) : " * string(totTimeAvg))
    println("\t -- flops/sec (avg) : " * string(flopsAvg/totTimeAvg))
    println("\t -- mems/sec (avg) (MB/s) : " * string(memsAvg/totTimeAvg))
end