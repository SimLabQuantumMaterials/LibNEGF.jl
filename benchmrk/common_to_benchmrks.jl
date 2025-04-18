# load the energy points from an input file
energValsIn = LibNEGF.load_energies("3x3")
EvalsIn = Vector{Float64}()
for (key, value) in energValsIn
    push!(EvalsIn, value)
end

# creating an artifial set of energy values, for benchmarking purposes
nrEvals = 5
eMin = EvalsIn[1]
eMax = last(EvalsIn)
Evals = Vector{Float64}(undef, nrEvals)
nrEvals > 1 ? deltaE = (eMax - eMin) / (nrEvals - 1) : deltaE = 0
for ix = 1:nrEvals
    Evals[ix] = eMin + (ix - 1) * deltaE
end

nrTotalThreads = 6
nrBLASthreads = 3
nrOuterThreads = nrTotalThreads / nrBLASthreads

LinearAlgebra.BLAS.set_num_threads(nrBLASthreads)

# list of the precisions to be tested
# ComplexF16 not fully functional in general
# precs = [ComplexF16, ComplexF32, ComplexF64]
if ARGS[1] == "cpu"
    precs = [ComplexF32, ComplexF64]
elseif ARGS[1] == "apple"
    precs = [ComplexF32]
end

# list of systems to loop over
# TODO : do we have to change this test to make use of a
#        different system?
systemNames = ["3x3"]

# list of k points
# TODO : do we want to have more than k=1 in this tests?
kpoints = [1]