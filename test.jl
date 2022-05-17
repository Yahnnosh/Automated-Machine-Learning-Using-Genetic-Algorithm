
using Distributions

activationfunctions = [x->max(x, 0), identity, tanh, sin]



struct Layer
    weights::Matrix{Float32}
    biases::Vector{Float32}
    activations::Vector{Function}
end

Base.show(io::IO, l::Layer) = print(io, l.activations)

Layer(input::Int, output::Int, std=1.0) = Layer(std.*randn(Float32, output, input), std.*randn(Float32, output), rand(activationfunctions, output))

function compute(a::Layer, in)
    map.(a.activations, a.biases .+ a.weights*in)
end

function crossover_mutate(a::Layer, b::Layer, std=0.01, p=0.01)
    idx = rand(Bool, length(a.biases))

    weights = similar(a.weights)
    weights[idx, :] = a.weights[idx, :]
    weights[.!idx, :] = b.weights[.!idx, :]

    biases = similar(a.biases)
    biases[idx] = a.biases[idx]
    biases[.!idx] = b.biases[.!idx]

    activations = similar(a.activations)
    activations[idx] = a.activations[idx]
    activations[.!idx] = b.activations[.!idx]

   
    weights = weights + std*randn(Float32, size(a.weights)...)
    biases = biases + std*randn(Float32, length(a.biases))

    idx = rand(Bernoulli(p), length(a.activations))

    newactivations = rand(activationfunctions, sum(idx))
    newweights = rand(Float32, sum(idx), size(a.weights)[2])
    newbiases = randn(Float32, sum(idx))

    activations[idx] = newactivations
    weights[idx, :] = newweights
    biases[idx] = newbiases

    Layer(weights, biases, activations)

end

struct Network
    layer1::Layer
    layer2::Layer
end

Network(input, output, hidden, std = 1.0) = Network(Layer(input, hidden, std), Layer(hidden, output, std))

function compute(a::Network, in)
    argmax.(eachcol(compute(a.layer2, compute(a.layer1, in)))).-1
end


function crossover_mutate(a::Network, b::Network, std=0.01, p=0.01)
    layer1 = crossover_mutate(a.layer1, b.layer1, std, p)
    layer2 = crossover_mutate(a.layer2, b.layer2, std, p)

    Network(layer1, layer2)
end



using MLDatasets

function evolve(batchsize, epochs, populationsize, reproducors, std=0.01, p=0.01)
    train_x, train_y = MNIST.traindata(Float32)
    train_x = reshape(train_x, 28*28, length(train_y))

    #test_x,  test_y  = MNIST.testdata(Float32)

    population = [Network(28*28, 32, 10) for _ in 1:populationsize]

    for nr in 1:epochs

        println("epoch: ", nr)
    
        idx = rand(1:length(train_y), batchsize)
        input = train_x[:, idx]
        output = train_y[idx]
    
        scores = [mapreduce(==, +, compute(organism, input), output) for organism in population]
    
        perm = sortperm(scores, rev=true)
    
        println("Best: ", scores[perm[1]])

        parentA = perm[rand(1:reproducors, populationsize)]
        parentB = perm[rand(1:reproducors, populationsize)]

        population = [crossover_mutate(population[a], population[b]) for (a, b) in zip(parentA, parentB)]
    end

    return population
end

popu = evolve(500, 100, 5, 3, 0.02, 0.1)

