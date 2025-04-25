Printf.@printf("Benchmarking bndiag_of_inv_direct(...)\n")

for systemx in systemNames
    for precx in precs
        for k in kpoints
            # load matrices and build M, hardcoding the energy
            # value for Sc to be 50 always, for benchmarking purposes
            listMatsToLoad = ["H", "S", "Sc"]
            loadedMats, blockSizes = load_matrices(systemx, 50, k,
                listMatsToLoad, precx)
            H = loadedMats[1]
            S = loadedMats[2]
            Se = loadedMats[3]
            # loading blockSizes only - this is redundant, but illustrates
            # that this can be done without any matrix loading
            listMatsToLoad = Vector{String}()
            loadedMats, blockSizes = load_matrices(systemx, 0, k,
                listMatsToLoad, precx)

            # create array of timers
            timers = []
            for ix = 1:Threads.nthreads()
                push!(timers, TimerOutput())
            end
            timerTagGlobal = "bndiag_of_inv_direct_" * string(precx)
            # do the inversions for all the energy points
            @timeit to timerTagGlobal Threads.@threads for E in Evals
                M = build_M_from_HS(H, S, Se, E)
                Printf.@printf(".")
                # get the block n-diagonal of M^-1 via inv(M)
                timerTagLocal = "bndiag_of_inv_direct_" * string(precx) * "_thread" * string(Threads.threadid())
                tid = Threads.threadid()
                @timeit timers[tid] timerTagLocal MInvNdiag = bndiag_of_inv_direct(M, blockSizes, Dict("in" => 3, "out" => 3))
            end
            for ix = 1:Threads.nthreads()
                merge!(to, timers[ix], tree_point=[timerTagGlobal])
            end
            Printf.@printf("\n")
        end
    end
end

Printf.@printf("\n")