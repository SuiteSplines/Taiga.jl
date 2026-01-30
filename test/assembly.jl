using UnivariateSplines

@testset "System matrix integrand evaluation" begin
    S = SplineSpace(1, Interval(0.0,1.0), 1)
    S1 = SplineSpace(1, Interval(0.0,1.0), 1)
    S2 = SplineSpace(2, Interval(0.0,1.0), 1)

    @test all(system_matrix_integrand(S,S,1,1; x=0.0) .== [1.0 0.0; 0.0 0.0])
    @test all(system_matrix_integrand(S,S,1,1; x=1.0) .== [0.0 0.0; 0.0 1.0])
    @test all(system_matrix_integrand(S,S,2,2; x=0.0) .== [1.0 -1.0; -1.0 1.0])
    @test all(system_matrix_integrand(S,S,2,2; x=1.0) .== [1.0 -1.0; -1.0 1.0])
    @test all(system_matrix_integrand(S,S,2,1; x=1.0) .== [0.0 0.0; -1.0 1.0])
    @test all(system_matrix_integrand(S,S,2,1; x=0.0) .== [-1.0 1.0; 0.0 0.0])
    @test all(system_matrix_integrand(S,S,1,2; x=1.0) .== [0.0 -1.0; 0.0 1.0])
    @test all(system_matrix_integrand(S,S,1,2; x=0.0) .== [-1.0 0.0; 1.0 0.0])
    @test size(system_matrix_integrand(S1,S2,1,1;x=0.0)) == (3,2)
    @test size(system_matrix_integrand(S1,S2,2,2;x=0.0)) == (3,2)
end