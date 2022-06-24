using Distributed
@everywhere using Flux
@everywhere using Flux.Optimise: update!

@everywhere using CUDA
@everywhere using CUDA: device!


@everywhere function test_train(device_id::Int)
    device!(device_id)
    x = rand(Float32, 10) |> cu
    model = Dense(10=>10) |> gpu
    opt = Descent(0.01)
    loss = x->x|>model|>sum

    for i in 1:1000    
        grads = gradient(() -> loss(x), Flux.params(model))
        update!(opt, Flux.params(model), grads) # update parameters
    end
    return loss(x)
end

pmap(test_train, [0, 1, 2, 3])