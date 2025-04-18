Printf.@printf("Benchmarking btrid_of_inv_direct(...)\n")

for systemx in systemNames
    for precx in precs
        for k in kpoints
            for E in Evals
                Printf.@printf(".")
                # load matrices and build T, hardcoding the energy
                # value for Sc to be 50 always, for benchmarking purposes
                listMatsToLoad = ["H", "S", "Sc"]
                loadedMats, blockSizes = load_matrices(systemx, 50, k,
                    listMatsToLoad, precx)
                H = loadedMats[1]
                S = loadedMats[2]
                Se = loadedMats[3]
                T = build_T_from_HS(H, S, Se, E)
                # loading blockSizes only - this is redundant, but illustrates
                # that this can be done without any matrix loading
                listMatsToLoad = Vector{String}()
                loadedMats, blockSizes = load_matrices(systemx, 0, k,
                    listMatsToLoad, precx)
                # get the block tridiagonal of T^-1 via inv(T)
                @timeit to "btrid_inv_direct_"*string(precx) TInvTrid = btrid_of_inv_direct(T, blockSizes)
            end
            Printf.@printf("\n")
        end
    end
end

Printf.@printf("\n")