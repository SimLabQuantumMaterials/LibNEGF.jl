Printf.@printf("Benchmarking bndiag_of_inv_direct(...)\n")

for systemx in systemNames
    for precx in precs
        for k in kpoints
            for E in Evals
                Printf.@printf(".")
                # load matrices and build M, hardcoding the energy
                # value for Sc to be 50 always, for benchmarking purposes
                listMatsToLoad = ["H", "S", "Sc"]
                loadedMats, blockSizes = load_matrices(systemx, 50, k,
                    listMatsToLoad, precx)
                H = loadedMats[1]
                S = loadedMats[2]
                Se = loadedMats[3]
                M = build_M_from_HS(H, S, Se, E)
                # loading blockSizes only - this is redundant, but illustrates
                # that this can be done without any matrix loading
                listMatsToLoad = Vector{String}()
                loadedMats, blockSizes = load_matrices(systemx, 0, k,
                    listMatsToLoad, precx)
                # get the block n-diagonal of M^-1 via inv(M)
                @timeit to "bndiag_inv_direct_"*string(precx) MInvNdiag = bndiag_of_inv_direct(M, blockSizes)
            end
            Printf.@printf("\n")
        end
    end
end

# enforce the garbage collector at this point
gc()

Printf.@printf("\n")