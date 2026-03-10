# 1 from disk, 2 is random
whereFrom = 2
# values for the synthetic matrix
npl = 720
blockSize = 64

useFinerTimings = Int(parse(Float64, ARGS[2]))

# enforcing benchmarks, for now, to happen on synthetic data only

# TODO : the following block needs to be restored if we want
# to go back to using whereFrom = 1
# # load the energy points from an input file
# if whereFrom == 1
#     energVals = LibNEGF.load_energies("3x3")
# else
#     energVals = Dict(50 => -0.16537196040135335, 250 => -0.0918733113340852, 150 => -0.12862263586771927)
# end
# Epoints = Vector{Int}()
# for (key, value) in energVals
#     push!(Epoints, key)
# end

# list of the precisions to be tested
# ComplexF16 not fully functional in general
# precs = [ComplexF16, ComplexF32, ComplexF64]
# at the moment, Metal does not support Float64,
# so replacing the precisions to test with
if ARGS[1] == "apple"
    precs = [ComplexF32]
elseif ARGS[1] == "cpu"
    precs = [ComplexF32, ComplexF64]
end

# # list of systems to loop over
# # do we have to change this test to make use of a different
# # system?
# systemNames = ["3x3"]

# list of k points
# do we want to have more than k=1 in this tests?
kpoints = [1]

# nrEPoints = size(Epoints)[1]
nrEPoints = 1