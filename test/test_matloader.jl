module TestLibNEGFMatloader

using LibNEGF, Test
import LinearAlgebra, Printf

@testset "Matloader" begin
    @testset "Matloader IO" begin
        # check that the read matrices are built properly with
        # the desired precision

        # load the energy points from an input file
        energVals = LibNEGF.loadEnergies("3x3");
        Epoints = Vector{Int}();
        for (key,value) in energVals
            push!(Epoints,key);
        end

        # list of the precisions to be tested
        precs = [ComplexF16,ComplexF32,ComplexF64];
        roundoffs = Dict{DataType,Float64}(ComplexF16=>1.0E-3,
                                           ComplexF32=>1.0E-7,
                                           ComplexF64=>1.0E-15);

        # list of systems to loop over
        # TODO : do we have to change this test to make use of a
        #        different system?
        systemNames = ["3x3"];

        # list of k points
        # TODO : do we want to have more than k=1 in this tests?
        kpoints = [1];

        for systemx in systemNames
            for E in Epoints
                for k in kpoints
                    for precx in precs
                        # list of matrices to load
                        listMatsToLoad = ["H","S","Sc","T"];
                        # first, load in F64
                        loadedMatsF64,blockSizes = loadMatrices(systemx,E,k,
                                                              listMatsToLoad,ComplexF64);
                        # then, in actual desired precision
                        loadedMats,blockSizes = loadMatrices(systemx,E,k,
                                                              listMatsToLoad,precx);
                        for ix in 1:size(listMatsToLoad)[1]
                            # now, check that the matrices loaded to the desired precision
                            # match with the F64 ones up to the corresponding accuracy. we
                            # use the Frobenius norm for this
                            MF64 = loadedMatsF64[ix];
                            M    = loadedMats[ix];
                            relErr = LinearAlgebra.norm(MF64-M,2)/LinearAlgebra.norm(MF64,2);
                            @test relErr < roundoffs[precx];
                        end
                    end
                end
            end
        end
    end

    # @testset "Matloader Consistent Matrices" begin
    #     # TODO : check here that the built T makes sense if directly
    #     #        loaded or built via S,H,\Sigma_{c}
    # end
end

end