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
    gemmFlops::Int
    gemmMems::Int
    mldivideFlops::Int
    mldivideMems::Int
    mrdivideFlops::Int
    mrdivideMems::Int
    luFlops::Int
    luMems::Int
    totalFlops::Int
    totalMems::Int
    nrCalls::Int
end

@ifdef "LIBNEGF_HW" begin
    if ENV["LIBNEGF_HW"] == "cpu"
        function flops_and_mems(whichKernel::String, cd::CountingData, sizes::Vector{Tuple{Int,Int}})
            if whichKernel == "_gemm"
                m = sizes[3][1]
                n = sizes[3][2]
                k = sizes[1][2]
                # fused multiply add not taken into account (should we multiply by 3 instead?)
                cd.gemmFlops += 6 * m * n * k
                cd.totalFlops += 6 * m * n * k
                cd.gemmMems += (k * (m + n) + 2 * m * n)
                cd.totalMems += (k * (m + n) + 2 * m * n)
            elseif whichKernel == "_mldivide"
                cd.mldivideFlops += 0
                cd.totalFlops += 0
                cd.mldivideMems += 0
                cd.totalMems += 0
            elseif whichKernel == "_mrdivide"
                cd.mrdivideFlops += 0
                cd.totalFlops += 0
                cd.mrdivideMems += 0
                cd.totalMems += 0
            elseif whichKernel == "_lu"
                cd.luFlops += 0
                cd.totalFlops += 0
                cd.luMems += 0
                cd.totalMems += 0
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
        flops_and_mems($(esc(suffx)), $(esc(cdx)), $(esc(sizesx)))
        @timeit ($(esc(tdx))).to ($(esc(tdx))).label * ($(esc(suffx))) $(esc(codex))
    end
end

function print_flops_and_mems_(cd::CountingData, to::TimerOutput, prec::DataType, suffx::String)
    nrCalls = cd.nrCalls
    if suffx == "gemm"
        # in gigaflops
        flopsAvg = (cd.gemmFlops * 1.0E-9) / nrCalls
        # data in GB
        memsAvg = ((sizeof(prec) * cd.gemmMems) / nrCalls) / (1024 * 1024 * 1024)
    elseif suffx == "lu"
        # in gigaflops
        flopsAvg = (cd.luFlops * 1.0E-9) / nrCalls
        # data in GB
        memsAvg = ((sizeof(prec) * cd.luMems) / nrCalls) / (1024 * 1024 * 1024)
    elseif suffx == "mldivide"
        # in gigaflops
        flopsAvg = (cd.mldivideFlops * 1.0E-9) / nrCalls
        # data in GB
        memsAvg = ((sizeof(prec) * cd.mldivideMems) / nrCalls) / (1024 * 1024 * 1024)
    elseif suffx == "mrdivide"
        # in gigaflops
        flopsAvg = (cd.mrdivideFlops * 1.0E-9) / nrCalls
        # data in GB
        memsAvg = ((sizeof(prec) * cd.mrdivideMems) / nrCalls) / (1024 * 1024 * 1024)
    elseif suffx == "total"
        # in gigaflops
        flopsAvg = (cd.totalFlops * 1.0E-9) / nrCalls
        # data in GB
        memsAvg = ((sizeof(prec) * cd.totalMems) / nrCalls) / (1024 * 1024 * 1024)
    end

    if suffx == "total"
        totTimeAvg = (TimerOutputs.time(to["bndiag_of_inv_ddrgf_"*string(prec)]["thread1_wo_first_total"]) * 1.0E-9) / nrCalls
    else
        totTimeAvg = (TimerOutputs.time(to["bndiag_of_inv_ddrgf_"*string(prec)]["thread1_wo_first_total"]["thread1_wo_first_"*suffx]) * 1.0E-9) / nrCalls
    end

    println("\t -- kernel : " * suffx)
    println("\t\t -- flops (avg) (megaflops) : " * string(flopsAvg))
    println("\t\t -- mems (avg) (GB) : " * string(memsAvg))
    println("\t\t -- time (avg) : " * string(totTimeAvg))
    println("\t\t -- flops/sec (avg) (GFLOPS) : " * string(flopsAvg / totTimeAvg))
    println("\t\t -- mems/sec (avg) (GB/s) : " * string(memsAvg / totTimeAvg))
end

function print_flops_and_mems(cd::CountingData, to::TimerOutput, prec::DataType)
    nrCalls = cd.nrCalls

    println("\nFlops and mems (" * string(prec) * "):")
    println("\t -- nr calls : " * string(nrCalls))
    print_flops_and_mems_(cd, to, prec, "gemm")
    print_flops_and_mems_(cd, to, prec, "lu")
    print_flops_and_mems_(cd, to, prec, "mldivide")
    print_flops_and_mems_(cd, to, prec, "mrdivide")
    print_flops_and_mems_(cd, to, prec, "total")
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