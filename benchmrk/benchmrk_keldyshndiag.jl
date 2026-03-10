Printf.@printf("Benchmarking keldyshndiag!(...)\n")

# choose the version of Keldysh's implementation to benchmark (see src/keldyshndiag.jl)
keldyshVersion = "v2"

# LinearAlgebra.BLAS.set_num_threads(Int(parse(Float64, ARGS[4])))

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

for precx in precs
    # check if there's enough memory for the allocations
    check_if_enough_mem_rkd(npl, blockSize, precx)

    # create a flops and mems counter for each precision and thread
    counters = Vector{CountingData}()
    if useFinerTimings == 1
        push!(counters, CountingData(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    else
        push!(counters, CountingData())
    end
    # create array of timers
    timers = Vector{TimerOutput}()
    push!(timers, TimerOutput())
    timerTagGlobal = "keldyshndiag_" * string(precx)

    for k in kpoints

        @timeit to timerTagGlobal begin
            # loop over bunches of energy points - fixed to 1 for now
            nrEgroups::Int = 1
            for iEG = 1:nrEgroups
                # preallocate large data per thread
                Mins = Vector{BlockMatrix}()
                Sns = Vector{BlockMatrix}()
                # if whereFrom == 1
                #     iE = ix + (iEG - 1) * Threads.nthreads()
                #     # load matrices and build M
                #     listMatsToLoad = ["H", "S", "Sc"]
                #     loadedMats, blockSizes = load_matrices(systemx, Epoints[iE], k,
                #         listMatsToLoad, precx)
                #     H = loadedMats[1]
                #     S = loadedMats[2]
                #     Se = loadedMats[3]
                #     Msp = build_M_from_HS(H, S, Se, energVals[Epoints[iE]])
                #     Min = bm_convert(Msp, blockSizes, Dict("in" => 3, "out" => 3))
                # else
                #     Min = bm_create_synthetic_random(npl, blockSize, precx, false)
                # end

                auxs = Vector{AuxDataKeldysh}()
                try
                    @time begin
                        Min = bm_create_synthetic_random(npl, blockSize, precx, false)
                        push!(Mins, Min)
                        Arandbm = bm_similar(Mins[1], 2)
                        Arandsp = bm_convert(Arandbm)
                        Arandsp = (Arandsp + Arandsp') / convert(precx, 2.0)
                        Sn = bm_convert(Arandsp, Arandbm.blockSizes, Arandbm.ndiag, true)
                        push!(Sns, Sn)
                        auxDataKeldysh = allocate_aux_data_Keldysh(Mins[1], Sns[1])
                        push!(auxs, auxDataKeldysh)
                    end
                catch OutOfMemoryError
                    error("The application tried to allocate beyond the available system memory")
                end

                # (?) force the garbage collector before doing the core computations
                GC.gc()

                tx(tId) = begin
                    if keldyshVersion == "v1"
                        ninvs = 1
                    else
                        ninvs = 10
                    end
                    # multiple inversions per energy point, for statistics purposes
                    for ix = 1:ninvs
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
                        @timeit timers[tId] timerTagLocalTotal keldyshndiag!(Mins[tId], Sns[tId], auxs[tId], td, cd)
                    end
                end

                tx(1)

                merge!(to, timers[1], tree_point=[timerTagGlobal])

                Printf.@printf("\n")
            end
        end
    end

    # print flops and mems counts for thread1 only
    if useFinerTimings == 1
        print_flops_and_mems(counters[1], to, precx, "keldyshndiag", true)
    end
end

# LinearAlgebra.BLAS.set_num_threads(1)

Printf.@printf("\n")