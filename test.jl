include("src/TensorNetworks.jl")


psi = randomMPS(2, 40, 10)
normalize!(psi)

sh = spinhalf()
