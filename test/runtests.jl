using LibNEGF, Test

@testset "LibNEGF.jl" begin
    @test LibNEGF.add(1,2)==3;
end

include("test_matloader.jl")