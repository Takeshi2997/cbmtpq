module Update
    include("./setup.jl")
    using .Const, LinearAlgebra

    function system(s, z)

        boltzmannfactor = exp.(-2.0 * s .* z)
        for ix in 1:Const.dimS
            if 1.0 > boltzmannfactor[ix]
                prob = rand(Float64)
                if prob < boltzmannfactor[ix]
                    s[ix] *= -1.0
                end
            else
                s[ix] *= -1.0
            end
        end

        return s
    end

    function bath(n, z)

        boltzmannfactor = exp.((1.0 .- 2.0 * n) .* z)
        for iy in 1:Const.dimB
            if 1.0 > boltzmannfactor[iy]
                prob = rand(Float64)
                if prob < boltzmannfactor[iy]
                    n[iy] = 1.0 - n[iy]
                end
            else
                n[iy] = 1.0 - n[iy]
            end
        end

        return n
    end
end
