Printf.@printf("Benchmarking bndiag_of_inv_direct(...)\n")

for systemx in systemNames
    for precx in precs
        for k in kpoints
            # create array of timers
            timers = Vector{TimerOutput}()
            for ix = 1:Threads.nthreads()
                push!(timers, TimerOutput())
            end
            timerTagGlobal = "bndiag_of_inv_direct_" * string(precx)

            # first, check if the number of threads divides the number of energy points,
            # exit if it doesn't
            # maybe put this check earlier ?
            if mod(size(Epoints)[1], Threads.nthreads()) != 0
                Printf.@printf("ERROR: the number of Julia threads (%d) does not divide the number \
                               of energy points (%d)\n", Threads.nthreads(), size(Epoints)[1])
                exit()
            end

            @timeit to timerTagGlobal begin
                # loading blockSizes only
                listMatsToLoad = Vector{String}()
                loadedMats, blockSizes = load_matrices(systemx, 0, k,
                    listMatsToLoad, precx)

                # loop over bunches of energy points
                nrEgroups::Int = size(Epoints)[1] / Threads.nthreads()
                for iEG = 1:nrEgroups
                    # preallocate large data per thread
                    Mins = Vector{BlockMatrix}()
                    Mouts = Vector{BlockMatrix}()
                    for ix = 1:Threads.nthreads()
                        iE = ix + (iEG - 1) * Threads.nthreads()
                        # load matrices and build M
                        listMatsToLoad = ["H", "S", "Sc"]
                        loadedMats, blockSizes = load_matrices(systemx, Epoints[iE], k,
                            listMatsToLoad, precx)
                        H = loadedMats[1]
                        S = loadedMats[2]
                        Se = loadedMats[3]
                        Msp = build_M_from_HS(H, S, Se, energVals[Epoints[iE]])
                        Min = convert_S2BM_ndiag(Msp, blockSizes, Dict("in" => 3, "out" => 3))
                        push!(Mins, Min)
                        push!(Mouts, copy_BM(Min))
                    end
                    auxs = Vector{AuxDataDDRGF}()
                    for ix = 1:Threads.nthreads()
                        push!(auxs, allocate_aux_data_DDRGF(Mins[ix]))
                    end

                    # (?) force the garbage collector before doing the core computations
                    GC.gc()

                    Threads.@threads for E in Epoints
                        tid = Threads.threadid()

                        ninvs = 10
                        # multiple inversions per energy point, for statistics purposes
                        for ix = 1:ninvs
                            if ix == 1
                                timerTagLocal = "thread" * string(Threads.threadid()) * "_first"
                            else
                                timerTagLocal = "thread" * string(Threads.threadid()) * "_wo_first"
                            end
                            Printf.@printf(".")
                            td = TimingData(timers[tid], timerTagLocal)
                            # TODO : we need to be able to pass <nothing> as the last parameter in the next function call
                            @timeit timers[tid] timerTagLocal*"_total" bndiag_of_inv_ddrgf!(Mouts[tid], Mins[tid], auxs[tid], td)
                        end
                    end

                    for ix = 1:Threads.nthreads()
                        merge!(to, timers[ix], tree_point=[timerTagGlobal])
                    end
                    Printf.@printf("\n")
                end
            end
        end
    end
end

Printf.@printf("\n")