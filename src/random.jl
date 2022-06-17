""" 
generate random number in cuda kernel, following the implementation of https://github.com/JuliaGPU/CUDA.jl/blob/master/src/random.jl

some discussions are in https://discourse.julialang.org/t/generating-random-number-from-inside-kernel/8071/9
"""

using CUDA
# using RandomNumbers.Xorshifts
using Random


function cpu_rand(num_samples)
    rng = Random.default_rng()
    rnd = map(1:num_samples) do x 
        # r = Xoroshiro128Plus()  # create a RNG with truly random seed.
        # return rand(r)
        return rand(rng)
    end
    return rnd 
end

@show cpu_rand(10)


mutable struct RNG <: AbstractRNG
    seed::UInt32
    counter::UInt32

    function RNG(seed::Integer)
        new(seed%UInt32, 0)
    end
end

make_seed() = Base.rand(RandomDevice(), UInt32)
RNG() = RNG(make_seed())

function rand_kernel(rand_out, num_samples::Int, seed::UInt32, counter::UInt32)
    device_rng = Random.default_rng()
    @inbounds Random.seed!(device_rng, seed, counter)
    i = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if i <= num_samples
        # r = Xoroshiro128Plus()  # create a RNG with truly random seed.
        rand_out[i] = rand(device_rng)
    end
    return nothing
end

function default_rng()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{rng::RNG}
    states = get!(task_local_storage(), :RNG) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        # CUDA RNG objects are cheap, so we don't need to cache them
        (; rng=RNG())
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.rng
end


function gpu_rand(num_samples)
    rnd = zeros(Float32, (num_samples, )) |> cu
    numblocks = ceil(Int, num_samples / 512)
    rng = default_rng()
    @cuda threads = 512 blocks = numblocks rand_kernel(rnd, num_samples, rng.seed, rng.counter)
    rnd = rnd |> Array
    return rnd
end

@show gpu_rand(10)