Printf.@printf("Benchmarking bndiag_of_inv_rgf_local! (Fused Approach)...\n")

# TODO : restore the following commented block if we want to go back
# to using whereFrom = 1
# first, check if the number of threads divides the number of energy points,
# exit if it doesn't
#if mod(nrEPoints, Threads.nthreads()) != 0
#    Printf.@printf("ERROR: the number of Julia threads (%d) does not divide the number \
#                    of energy points (%d)\n", Threads.nthreads(), nrEPoints)
#    exit()
#end

# NOTE : RGF should be run with one Julia thread, and possible multiple
# BLAS threads

if Threads.nthreads() != 1
    error("The number of Julia threads when running RGF must be 1")
end

# Set the original bandwidth for the general n-diagonal matrix
nDiagVal = 5
ndiagDict = Dict("in" => nDiagVal, "out" => nDiagVal)

for precx in precs
    # # check if there's enough memory for the allocations
    # check_if_enough_mem_rgf(npl, blockSize, precx)

    # create a flops and mems counter for each precision
    counters = Vector{CountingData}()
    if useFinerTimings == 1
        push!(counters, CountingData(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    else
        push!(counters, CountingData())
    end
    # create array of timers
    timers = Vector{TimerOutput}()
    push!(timers, TimerOutput())
    timerTagGlobal = "bndiag_of_inv_fused_rgf_" * string(precx)

    for k in kpoints
        @timeit to timerTagGlobal begin
            # loop over bunches of energy points - fixed to 1 for now
            nrEgroups::Int = 1
            for iEG = 1:nrEgroups
                # preallocate large data per thread
                Mins = Vector{BlockMatrix}()
                Mouts = Vector{BlockMatrix}()

                auxs = Vector{AuxDataRGF}()
                try
                    begin
                        # 1. Generate the synthetic n-diagonal matrix
                        MinNDiag = bm_create_synthetic_random(npl, blockSize, precx, false, ndiagDict)

                        # 2. Fuse the blocks to recast it as a block tridiagonal matrix
                        # (Assumes bm_fuse_to_tridiagonal is available in your loaded modules)
                        Min = bm_fuse_to_tridiagonal(MinNDiag)

                        push!(Mins, Min)
                        push!(Mouts, bm_copy(Min))

                        # 3. Allocate standard RGF aux data for the new tridiagonal matrix
                        push!(auxs, allocate_aux_data_RGF(Mins[1]))
                    end
                catch e
                    if e isa OutOfMemoryError
                        # TODO : handle this better, but perhaps a suggestion in params change
                        error("The application tried to allocate beyond the available system memory")
                    else
                        rethrow(e)
                    end
                end

                # (?) force the garbage collector before doing the core computations
                GC.gc()

                tx(tId) = begin
                    ninvs = 10
                    # multiple inversions per energy point, for statistics purposes
                    for ix = 1:ninvs
                        # do a clear separation when timing the first inversion vs the others
                        if ix == 1
                            timerTagLocal = "thread" * string(tId) * "_first"
                        else
                            timerTagLocal = "thread" * string(tId) * "_wo_first"
                        end
                        # don't include the first inversion in the flops and mems counting
                        if useFinerTimings == 1
                            if ix == 1
                                cd = CountingData(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                            else
                                cd = counters[tId]
                                cd.nrCalls += 1
                            end
                        else
                            cd = CountingData()
                        end

                        Printf.@printf(".")
                        if useFinerTimings == 1
                            td = TimingData(timers[tId], timerTagLocal)
                        else
                            td = TimingData()
                        end
                        timerTagLocalTotal = timerTagLocal * "_total"

                        # Call the STANDARD RGF computation on the fused matrix
                        @timeit timers[tId] timerTagLocalTotal bndiag_of_inv_rgf_local!(Mouts[tId], Mins[tId], auxs[tId], td, cd)
                    end
                end

                tx(1)

                Printf.@printf("\n")
            end
        end
    end

    merge!(to, timers[1], tree_point=[timerTagGlobal])

    # print flops and mems counts for thread1 only
    if useFinerTimings == 1
        print_flops_and_mems(counters[1], to, precx, "bndiag_of_inv_fused_rgf", true)
    end
end

Printf.@printf("\n")