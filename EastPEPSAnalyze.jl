using HDF5
using DelimitedFiles
include("src/TensorNetworks.jl")
direct = "D:/East Data/2d/"
Ns = [6, 10, 14, 18, 22, 26, 30]
Ns = [30]
c = 0.5
ss = -exp10.(range(-3, stop=0, length=61))
ss2 = exp10.(range(-5, stop=1, length=121))
append!(ss, ss2)
push!(ss, 0.0)
unique!(ss)
sort!(ss)
ss = round.(ss, digits=10)
for N = Ns
    thetas = []
    activities = []
    for s = ss
        directfile = string(direct, "PEPS/c = ", c, "/N = ", N, "/s = ", s, ".h5")
        f = h5open(directfile)
        push!(thetas, real(read(f, "scgf")))
        push!(activities, real(read(f, "activity")))
        close(f)
    end

    writedlm(string(direct, "LDs/c = ", c, "/N = ", N, ".csv"), [ss, thetas, activities], ",")
end
