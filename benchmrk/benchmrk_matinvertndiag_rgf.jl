Printf.@printf("Benchmarking bndiag_of_inv_ddrgf!(...)\n")

LinearAlgebra.BLAS.set_num_threads(Int(parse(Float64, ARGS[5])))

# first, check if the number of threads divides the number of energy points,
# exit if it doesn't
if mod(nrEPoints, Threads.nthreads()) != 0
    Printf.@printf("ERROR: the number of Julia threads (%d) does not divide the number \
                    of energy points (%d)\n", Threads.nthreads(), nrEPoints)
    exit()
end

for systemx in systemNames
    for precx in precs
        # create a flops and mems counter for each precision and thread
        counters = Vector{CountingData}()
        for ix = 1:Threads.nthreads()
            if useFinerTimings == 1
                push!(counters, CountingData(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            else
                push!(counters, CountingData())
            end
        end
        # create array of timers
        timers = Vector{TimerOutput}()
        for ix = 1:Threads.nthreads()
            push!(timers, TimerOutput())
        end
        timerTagGlobal = "bndiag_of_inv_ddrgf_" * string(precx)

        for k in kpoints
            @timeit to timerTagGlobal begin
                # loop over bunches of energy points
                nrEgroups::Int = size(Epoints)[1] / Threads.nthreads()
                for iEG = 1:nrEgroups
                    # preallocate large data per thread
                    Mins = Vector{BlockMatrix}()
                    Mouts = Vector{BlockMatrix}()
                    for ix = 1:Threads.nthreads()
                        if whereFrom == 1
                            iE = ix + (iEG - 1) * Threads.nthreads()
                            # load matrices and build M
                            listMatsToLoad = ["H", "S", "Sc"]
                            loadedMats, blockSizes = load_matrices(systemx, Epoints[iE], k,
                                listMatsToLoad, precx, whereFrom)
                            H = loadedMats[1]
                            S = loadedMats[2]
                            Se = loadedMats[3]
                            Msp = build_M_from_HS(H, S, Se, energVals[Epoints[iE]])
                            Min = bm_convert(Msp, blockSizes, Dict("in" => 3, "out" => 3))
                        else
                            Min = bm_create_synthetic_random(10, 648, precx)
                        end
                        push!(Mins, Min)
                        push!(Mouts, bm_copy(Min))
                    end
                    auxs = Vector{AuxDataRGF}()
                    for ix = 1:Threads.nthreads()
                        push!(auxs, allocate_aux_data_RGF(Mins[ix], parse(Int, ARGS[4]), parse(Int, ARGS[5])))
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
                            @timeit timers[tId] timerTagLocalTotal bndiag_of_inv_rgf_local!(Mouts[tId], Mins[tId], auxs[tId], td, cd)
                        end
                    end

                    Threads.@threads for ix in 1:Threads.nthreads()
                        tx(ix)
                    end

                    Printf.@printf("\n")
                end
            end
        end

        for ix = 1:Threads.nthreads()
            merge!(to, timers[ix], tree_point=[timerTagGlobal])
        end

        # print flops and mems counts for thread1 only
        if useFinerTimings == 1
            print_flops_and_mems(counters[1], to, precx, "bndiag_of_inv_ddrgf", true)
        end
    end
end

LinearAlgebra.BLAS.set_num_threads(1)

Printf.@printf("\n")