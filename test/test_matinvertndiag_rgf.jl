# # computing the block n-diagonal of the inverse of T
# # by the DD-RGF method

# include("common_to_test.jl")

# for systemx in systemNames
#     for E in Epoints
#         for k in kpoints
#             # pre-compute the condition number in double precision
#             # list of matrices to load
#             listMatsToLoad = ["H", "S", "Sc"]
#             loadedMats, blockSizes = load_matrices(systemx, E, k,
#                 listMatsToLoad, ComplexF64)
#             H = loadedMats[1]
#             S = loadedMats[2]
#             Se = loadedMats[3]
#             M = build_M_from_HS(H, S, Se, energVals[E])
#             # LA.cond(..) makes use of LA.opnorm(..)
#             condNum = LinearAlgebra.cond(Array(M))

#             for precx in precs
#                 # load matrices and build M
#                 listMatsToLoad = ["H", "S", "Sc"]
#                 loadedMats, blockSizes = load_matrices(systemx, E, k,
#                     listMatsToLoad, precx)
#                 H = loadedMats[1]
#                 S = loadedMats[2]
#                 Se = loadedMats[3]
#                 M = build_M_from_HS(H, S, Se, energVals[E])

#                 # load Gr
#                 listMatsToLoad = ["Gr"]
#                 loadedMats, blockSizes = load_matrices(systemx, E, k,
#                     listMatsToLoad, precx)
#                 Gr = loadedMats[1]

#                 # loading blockSizes only - this is redundant, but illustrates
#                 # that this can be done without any matrix loading
#                 listMatsToLoad = Vector{String}()
#                 loadedMats, blockSizes = load_matrices(systemx, E, k,
#                     listMatsToLoad, precx)

#                 # convert to BlockMatrix
#                 Mbm = convert_S2BM_ndiag(M, blockSizes)
#                 # get the block n-diagonal of M^-1 via DD-RGF
#                 MbmInvNdiag = bndiag_of_inv_dd_rgf(M, blockSizes, Dict("in" => 3, "out" => 3))
#                 # convert back to sparse
#                 MinvSp = convert_BM2S_trid(MbmInvNdiag)

#                 relErr = LinearAlgebra.opnorm(Array(MinvSp - Gr), 2) / LinearAlgebra.opnorm(Array(Gr), 2)
#                 # making a rough assumption on backward stability. The additional
#                 # 1.0E1 is because we see a loss in 1 digit in some cases
#                 @test relErr < roundoffs[precx] * condNum * 1.0E1
#             end
#         end
#     end
# end