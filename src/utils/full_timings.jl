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
        function flops_and_mems(whichKernel::String, cd::CountingData, sizes::Vector{Tuple{Int, Int}})
            if whichKernel == "_gemm"
                m = sizes[3][1]
                n = sizes[3][2]
                k = sizes[1][2]
                # fused multiply add not taken into account
                cd.flops += 6*m*n*k
                cd.mems += ( k*(m+n) + 2*m*n )
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
    # in gigaflops
    flopsAvg = (cd.flops*1.0E-9)/nrCalls
    # data in GB
    memsAvg = ((sizeof(prec)*cd.mems)/nrCalls)/(1024*1024*1024)
    totTimeAvg = ( TimerOutputs.time(to["bndiag_of_inv_ddrgf_"*string(prec)]["thread1_wo_first_total"]["thread1_wo_first_gemm"]) * 1.0E-9 ) / nrCalls

    println("\nFlops and mems (" * string(prec) * "):")
    println("\t -- nr calls : " * string(nrCalls))
    println("\t -- flops (avg) (megaflops) : " * string(flopsAvg))
    println("\t -- mems (avg) (GB) : " * string(memsAvg))
    println("\t -- time (avg) : " * string(totTimeAvg))
    println("\t -- flops/sec (avg) (GFLOPS) : " * string(flopsAvg/totTimeAvg))
    println("\t -- mems/sec (avg) (GB/s) : " * string(memsAvg/totTimeAvg))
end

"""

on a MacBook Air with M3 chip:

-- first, note that in our 3x3 runs the arithmetic intensity of GEMM is:

   ( 6*648*648*648 ) / ( 8 * (648*(648+648)+2*648*648) ) = 121.5

   with complex double

-- now, the CPUs in the M3 chip have the following properties:

   * RAM bandwidth : 100 GB/s
   * single core : 25.58 GFLOPS
   * so, its arithmethic intensity : 25.58 / 100 = 0.26

"""