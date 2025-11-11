Printf.@printf("Benchmarking bndiag_of_inv_ddrgf!(...)\n")

# first, check if the number of threads divides the number of energy points,
# exit if it doesn't
if mod(nrEPoints, 1) != 0
    Printf.@printf("ERROR: the number of Julia threads (%d) does not divide the number \
                    of energy points (%d)\n", 1, nrEPoints)
    exit()
end

for systemx in systemNames
    for precx in precs
        # create a flops and mems counter for each precision and thread
        counters = Vector{CountingData}()
        for ix = 1:1
            if useFinerTimings == 1
                push!(counters, CountingData(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            else
                push!(counters, CountingData())
            end
        end
        # create array of timers
        timers = Vector{TimerOutput}()
        for ix = 1:1
            push!(timers, TimerOutput())
        end
        timerTagGlobal = "bndiag_of_inv_ddrgf_" * string(precx)

        # (?) force the garbage collector before doing the core computations
        GC.gc()

        for k in kpoints
            # loop over bunches of energy points
            nrEgroups::Int = size(Epoints)[1] / 1
            for iEG = 1:nrEgroups
                # preallocate large data per thread
                Mins = Vector{BlockMatrix}()
                Mouts = Vector{BlockMatrix}()
                for ix = 1:1
                    if whereFrom == 1
                        iE = ix + (iEG - 1) * 1
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
                        Min = bm_create_synthetic_random(npl, blockSize, precx)
                    end
                    push!(Mins, Min)
                    push!(Mouts, bm_copy(Min))
                end

                # pre-allocate buffer data for parallel RGF
                if useFinerTimings == 1
                    cdSetup = CountingData(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                else
                    cdSetup = CountingData()
                end
                if useFinerTimings == 1
                    tdSetup = TimingData(TimerOutput(), "setup")
                else
                    tdSetup = TimingData()
                end
                listOfAuxDataPar = allocate_aux_data_DDRGF(Mins[1], false, parse(Int, ARGS[3]), parse(Int, ARGS[4]),
                    tdSetup, cdSetup)

                listOfListOfAuxDataPar = Vector{Vector{AuxDataDDRGF}}()
                push!(listOfListOfAuxDataPar, listOfAuxDataPar)

                @timeit to timerTagGlobal begin

                    tx(tId) = begin
                        ninvs = 10
                        # multiple inversions per energy point, for statistics purposes
                        for ix = 1:ninvs
                            # do a clear separation when timing the first inversion vs the others
                            if ix == 1
                                timerTagLocal = "first"
                            else
                                timerTagLocal = "wo_first"
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
                            LinearAlgebra.BLAS.set_num_threads(listOfListOfAuxDataPar[tId][1].nrBLASThreadsInner)
                            begin
                                @timeit timers[tId] timerTagLocalTotal bndiag_of_inv_ddrgf!(Mins[tId], listOfListOfAuxDataPar[tId], td, cd, 1)
                                bm_copy!(Mouts[tId], listOfListOfAuxDataPar[tId][1].buffMout)
                            end
                        end
                    end

                    tx(1)

                    Printf.@printf("\n")
                end
            end
        end

        for ix = 1:1
            merge!(to, timers[ix], tree_point=[timerTagGlobal])
        end

        # print flops and mems counts for thread1 only
        if useFinerTimings == 1
            print_flops_and_mems(counters[1], to, precx, "bndiag_of_inv_ddrgf", false)
        end
    end
end

Printf.@printf("\n")