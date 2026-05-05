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
    # otherMems::Int
    totalFlops::Int
    totalMems::Int
    nrCalls::Int
    # sizes::Vector{Int}
end

@ifdef "LIBNEGF_HW" begin
    if ENV["LIBNEGF_HW"] == "cpu"
        function flops_and_mems(whichKernel::String, cd::CountingData, A, B, C)
            # TODO : carefully restore the following (avoiding, e.g., InexactError)
            # n::Int = 0
            # m::Int = 0
            # k::Int = 0
            # if whichKernel == "_gemm"
            #     m = size(C)[1]
            #     n = size(C)[2]
            #     k = size(A)[2]
            #     # 2 for complex add and 6 for complex mult?
            #     cd.gemmFlops += (2 + 6) * m * n * k
            #     cd.totalFlops += (2 + 6) * m * n * k
            #     cd.gemmMems += 2 * (k * (m + n) + 2 * m * n)
            #     cd.totalMems += 2 * (k * (m + n) + 2 * m * n)
            # elseif whichKernel == "_mldivide"
            #     n = size(A)[1]
            #     m = size(A)[2]
            #     # 2 for complex add/sub and 6 for complex mult/div?
            #     numAdds::Int = 2 * (n * n - n * (n + 1) / 2 - n)
            #     numMuls::Int = 6 * (n * n - n * (n + 1) / 2)
            #     numDivs::Int = 6 * n
            #     numSubs::Int = 2 * n
            #     cd.mldivideFlops += 2 * m * (numAdds + numMuls + numDivs + numSubs)
            #     cd.totalFlops += 2 * m * (numAdds + numMuls + numDivs + numSubs)
            #     cd.mldivideMems += 0
            #     cd.totalMems += 0
            # elseif whichKernel == "_mrdivide"
            #     cd.mrdivideFlops += 0
            #     cd.totalFlops += 0
            #     cd.mrdivideMems += 0
            #     cd.totalMems += 0
            # elseif whichKernel == "_lu"
            #     n = size(A)[1]
            #     # 2 for complex add and 6 for complex mult?
            #     println(n * n * n)
            #     exit()
            #     cd.luFlops += ((2 + 6) / 3) * n * n * n
            #     cd.totalFlops += ((2 + 6) / 3) * n * n * n
            #     cd.luMems += 0
            #     cd.totalMems += 0
            # end
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
macro timewrap(tdx, suffx, codex)
    return quote
        @timeit ($(esc(tdx))).to ($(esc(tdx))).label * ($(esc(suffx))) $(esc(codex))
    end
end

macro countwrap(cdx, suffx, A, B, C, codex)
    return quote
        flops_and_mems($(esc(suffx)), $(esc(cdx)), $(esc(A)), $(esc(B)), $(esc(C)))
        $(esc(codex))
    end
end

function print_flops_and_mems_(cd::CountingData, to::TimerOutput, prec::DataType, suffx::String, method::String,
    isSeq::Bool)
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
    # elseif suffx == "other"
    #     # in gigaflops
    #     flopsAvg = (cd.otherFlops * 1.0E-9) / nrCalls
    #     # data in GB
    #     memsAvg = ((sizeof(prec) * cd.otherMems) / nrCalls) / (1024 * 1024 * 1024)
    elseif suffx == "total"
        # in gigaflops
        flopsAvg = (cd.totalFlops * 1.0E-9) / nrCalls
        # data in GB
        memsAvg = ((sizeof(prec) * cd.totalMems) / nrCalls) / (1024 * 1024 * 1024)
    end

    if suffx == "total"
        if isSeq == true
            totTimeAvg = (TimerOutputs.time(to[method*"_"*string(prec)]["thread1_wo_first_total"]) * 1.0E-9) / nrCalls
        else
            totTimeAvg = (TimerOutputs.time(to[method*"_"*string(prec)]["wo_first_total"]) * 1.0E-9) / nrCalls
        end
    else
        if isSeq == true
            totTimeAvg = (TimerOutputs.time(to[method*"_"*string(prec)]["thread1_wo_first_total"]["thread1_wo_first_"*suffx]) * 1.0E-9) / nrCalls
        else
            totTimeAvg = 0
            # avoiding KeyError with try/catch. This could be done better
            try
                totTimeAvg += (TimerOutputs.time(to[method*"_"*string(prec)]["wo_first_total"]["wo_first_T11inv"]["wo_first_"*suffx]) * 1.0E-9) / nrCalls
            catch e
            end
            try
                totTimeAvg += (TimerOutputs.time(to[method*"_"*string(prec)]["wo_first_total"]["wo_first_SCinv"]["wo_first_"*suffx]) * 1.0E-9) / nrCalls
            catch e
            end
            try
                totTimeAvg += (TimerOutputs.time(to[method*"_"*string(prec)]["wo_first_total"]["wo_first_SCinv"]["wo_first_SeqInv"]["wo_first_"*suffx]) * 1.0E-9) / nrCalls
            catch e
            end
            try
                totTimeAvg += (TimerOutputs.time(to[method*"_"*string(prec)]["wo_first_total"]["wo_first_Hopp12"]["wo_first_"*suffx]) * 1.0E-9) / nrCalls
            catch e
            end
            try
                totTimeAvg += (TimerOutputs.time(to[method*"_"*string(prec)]["wo_first_total"]["wo_first_Hopp21"]["wo_first_"*suffx]) * 1.0E-9) / nrCalls
            catch e
            end
            try
                totTimeAvg += (TimerOutputs.time(to[method*"_"*string(prec)]["wo_first_total"]["wo_first_11"]["wo_first_"*suffx]) * 1.0E-9) / nrCalls
            catch e
            end
        end
    end

    println(Core.stdout, "\t -- kernel : " * suffx)
    println(Core.stdout, "\t\t -- flops (avg) (megaflops) : " * string(flopsAvg))
    println(Core.stdout, "\t\t -- mems (avg) (GB) : " * string(memsAvg))
    println(Core.stdout, "\t\t -- time (avg) : " * string(totTimeAvg))
    println(Core.stdout, "\t\t -- flops/sec (avg) (GFLOPS) : " * string(flopsAvg / totTimeAvg))
    println(Core.stdout, "\t\t -- mems/sec (avg) (GB/s) : " * string(memsAvg / totTimeAvg))
end

function print_flops_and_mems(cd::CountingData, to::TimerOutput, prec::DataType, method::String,
    isSeq::Bool)
    nrCalls = cd.nrCalls

    println(Core.stdout, "\nFlops and mems (" * string(prec) * " - master thread only):")
    println(Core.stdout, "\t -- nr calls : " * string(nrCalls))
    print_flops_and_mems_(cd, to, prec, "gemm", method, isSeq)
    print_flops_and_mems_(cd, to, prec, "lu", method, isSeq)
    print_flops_and_mems_(cd, to, prec, "mldivide", method, isSeq)
    print_flops_and_mems_(cd, to, prec, "total", method, isSeq)
end

function print_flops_and_mems_si(cd::CountingData, to::TimerOutput, prec::DataType, method::String,
    isSeq::Bool)
    nrCalls = cd.nrCalls

    println("\nFlops and mems (" * string(prec) * " - master thread only):")
    println("\t -- nr calls : " * string(nrCalls))
    print_flops_and_mems_(cd, to, prec, "gemm", method, isSeq)
    # print_flops_and_mems_(cd, to, prec, "other", method, isSeq)
    # print_flops_and_mems_(cd, to, prec, "lu", method, isSeq)
    print_flops_and_mems_(cd, to, prec, "mldivide", method, isSeq)
    print_flops_and_mems_(cd, to, prec, "total", method, isSeq)
end