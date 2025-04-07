# computing the direct inverse via inv(...), this test
# is added here for completeness, and as a base for the
# subsequent ones

# load the energy points from an input file
energVals = LibNEGF.loadEnergies("3x3");
Epoints = Vector{Int}();
for (key, value) in energVals
    push!(Epoints, key)
end

# list of the precisions to be tested
# things not working well with ComplexF16
# precs = [ComplexF16,ComplexF32,ComplexF64];
precs = [ComplexF32, ComplexF64];
roundoffs = Dict{DataType,Float64}(ComplexF16 => 1.0E-3,
    ComplexF32 => 1.0E-7,
    ComplexF64 => 1.0E-15);

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
                listMatsToLoad = ["H", "S", "Sc"]
                # then, in actual desired precision
                loadedMats, blockSizes = loadMatrices(systemx, E, k,
                    listMatsToLoad, precx)
                H = loadedMats[1]
                S = loadedMats[2]
                Sc = loadedMats[3]
                # the convert(...) in the following line is to avoid casting
                # to ComplexF64
                T = convert(precx, energVals[E]) * S - H - Sc
                Tdense = Array(T)
                TdenseInv = inv(Tdense)
                relErr = LinearAlgebra.norm(Tdense * TdenseInv - LinearAlgebra.I, 2) / LinearAlgebra.norm(Tdense, 2)
                # we are hardcoding this value of 1.0E3 here, as we know
                # that the conditioning of Tdense is around 1.0E3 for the
                # test matrices at hand
                @test relErr < roundoffs[precx] * 1.0E3
            end
        end
    end
end