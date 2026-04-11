module TestLibNEGFBackend

using LibNEGF, Test, Random
import LinearAlgebra

# fix seed to have reproducible tests
Random.seed!(1234)

# TODO : documentation
# the two blocks passed live on the device
function check_if_equal(B1, B2, roundoff, relxFctr::Float64)
    B1cpu = be_copy_from_hw(B1)
    B2cpu = be_copy_from_hw(B2)
    normDen = LinearAlgebra.norm(B2cpu, 2)
    if normDen == 0
        relErr = LinearAlgebra.norm(B1cpu - B2cpu, 2)
    else
        relErr = LinearAlgebra.norm(B1cpu - B2cpu, 2) / normDen
    end
    @test relErr < (roundoff * relxFctr)
end

@testset "Backend" begin
    @testset "Backend Device Transfers" begin
        # check that moving data from, to and within the selected
        # hardware device works correctly

        include("common_to_test.jl")

        # we only want to load one matrix, but we keep the for loops
        # for generality and possible extended use in the future
        for k in [kpoints[1]]
            for precx in precs
                # if whereFrom == 1
                #     # list of matrices to load
                #     listMatsToLoad = ["H", "S", "Sc"]
                #     # load in the desired precision
                #     loadedMats, blockSizes = load_matrices(systemx, E, k,
                #         listMatsToLoad, precx, whereFrom)
                #     H = loadedMats[1]
                #     S = loadedMats[2]
                #     Se = loadedMats[3]
                #     M = build_M_from_HS(H, S, Se, energVals[E])

                #     # convert to BlockMatrix, this lives in the device always
                #     Abm = bm_convert(M, blockSizes, Dict("in" => 3, "out" => 3))
                # else
                #     Abm = bm_create_synthetic_random(10, 64, precx, false)
                # end

                Abm = bm_create_synthetic_random(10, 64, precx, false)

                # performance is not a problem here, therefore we call the garbage
                # collector after each test to make sure there are no memory issues
                # being missed

                td = TimingData()
                cd = CountingData()

                Bz = be_zero_array(precx, size(Abm.M[1, 1]))
                be_copy_in_hw!(Bz, Abm.M[1, 1])
                check_if_equal(Bz, Abm.M[1, 1], roundoffs[precx], 1.0E0)
                Bz = 0
                GC.gc()

                Bz = be_copy_in_hw(Abm.M[1, 1])
                check_if_equal(Bz, Abm.M[1, 1], roundoffs[precx], 1.0E0)
                Bz = 0
                GC.gc()

                Bz = be_zero_array(precx, size(Abm.M[1, 1]))
                Bx = be_copy_to_hw(zeros(precx, size(Abm.M[1, 1])))
                check_if_equal(Bz, Bx, roundoffs[precx], 1.0E0)
                Bz = 0
                Bx = 0
                GC.gc()

                Bzcpu = zeros(precx, size(Abm.M[1, 1]))
                Bzdev = be_zero_array(precx, size(Abm.M[1, 1]))
                be_copy_from_hw!(Bzcpu, Abm.M[1, 1])
                be_copy_to_hw!(Bzdev, Bzcpu)
                check_if_equal(Bzdev, Abm.M[1, 1], roundoffs[precx], 1.0E0)
                Bzcpu = 0
                Bzdev = 0
                GC.gc()

                Bzlu = be_zero_lu(precx, size(Abm.M[1, 1])[1])
                be_lu!(Bzlu, Abm.M[1, 1], td, cd)
                Aflu = be_A_from_LU(Bzlu)
                check_if_equal(Aflu, Abm.M[1, 1], roundoffs[precx], 1.0E1)
                Bzlu = 0
                Aflu = 0
                GC.gc()

                Bzlu = be_lu(Abm.M[1, 1], td, cd)
                Aflu = be_A_from_LU(Bzlu)
                check_if_equal(Aflu, Abm.M[1, 1], roundoffs[precx], 1.0E1)
                Bzlu = 0
                Aflu = 0
                GC.gc()

                B1 = be_copy_in_hw(Abm.M[1, 1])
                B2 = be_copy_in_hw(Abm.M[1, 2])
                B1cpu = be_copy_from_hw(B1)
                B2cpu = be_copy_from_hw(B2)
                B3cpu = B1cpu * B2cpu
                B3 = be_copy_to_hw(B3cpu)
                B3x = be_mul(B1, B2)
                check_if_equal(B3x, B3, roundoffs[precx], 1.0E0)
                B1 = 0
                B2 = 0
                B1cpu = 0
                B2cpu = 0
                B3cpu = 0
                B3 = 0
                B3x = 0
                GC.gc()

                B = be_copy_in_hw(Abm.M[1, 2])
                Bz = be_zero_array(precx, size(Abm.M[1, 2]))
                Blu = be_lu(Abm.M[1, 1], td, cd)
                be_mldivide!('N', Bz, B, Blu, td, cd)
                Bout = be_mul(Abm.M[1, 1], Bz)
                check_if_equal(Bout, B, roundoffs[precx], 1.0E3)
                B = 0
                Bz = 0
                Blu = 0
                Bout = 0
                GC.gc()

                A = be_copy_in_hw(Abm.M[1, 1])
                B = be_copy_in_hw(Abm.M[1, 2])
                C = be_copy_in_hw(B)
                Acpu = be_copy_from_hw(A)
                Bcpu = be_copy_from_hw(B)
                Ccpu = be_copy_from_hw(C)
                Ccpu = Ccpu - Acpu * Bcpu
                be_gemm!('N', 'N', convert(precx, -1.0), A, B, convert(precx, 1.0), C, td, cd)
                Cx = be_copy_to_hw(Ccpu)
                # the 1.0E1 is for some roundings, mostly due to GPUs
                check_if_equal(C, Cx, roundoffs[precx], 1.0E1)
                A = 0
                B = 0
                C = 0
                Acpu = 0
                Bcpu = 0
                Ccpu = 0
                Cx = 0
                GC.gc()

                Alu = be_lu(Abm.M[1, 1], td, cd)
                Ainv = be_copy_in_hw(Abm.M[1, 1])
                be_inv_from_lu!(Ainv, Alu)
                idM = be_identity(precx, size(Abm.M[1, 1])[1])
                idMx = be_mul(Abm.M[1, 1], Ainv)
                if precx == ComplexF32
                    relxFctr = 1.0E4
                elseif precx == ComplexF64
                    relxFctr = 1.0E3
                end
                check_if_equal(idM, idMx, roundoffs[precx], relxFctr)
                Alu = 0
                Ainv = 0
                idM = 0
                idMx = 0
                GC.gc()
            end
        end
    end
end

end