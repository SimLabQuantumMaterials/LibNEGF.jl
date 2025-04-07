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