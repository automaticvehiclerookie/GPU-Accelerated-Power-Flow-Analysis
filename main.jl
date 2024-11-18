"""
    Main function for the AC power flow
"""

# Detect the current working operating system
if Sys.iswindows()
    # Add the path to the data folder
    push!(LOAD_PATH, pwd()*"\\src\\")
    include(pwd()*"\\data\\case118.jl")
else
    # Add the path to the data folder
    push!(LOAD_PATH, pwd()*"/src/")
    include(pwd()*"/data/case118.jl")
    using AppleAccelerate
end
# push!(LOAD_PATH, pwd()*"\\data\\");
using PowerFlow
using MATLAB
using Plots
# using BenchmarkTools
# Start a MATLAB engine
# mat"addpath('C:\\Codes\\matpower')"
# mpc = case118();
opt = PowerFlow.options() # The initial settings 
opt["PF"]["NR_ALG"] = "gpu";
opt["PF"]["ENFORCE_Q_LIMS"]=0
mat"addpath('C:/Users/13733/Desktop/matpower7.1/data')"
mpc = mat"case300";

# mpc = mat"case_ieee30";
# (1) Run the power flow from Julia 
# Record the running time
# We need a conversion from external format to internal format

@time mpc = PowerFlow.runpf(mpc, opt);
# @code_warntype  PowerFlow.runpf(mpc, opt);
# using Serialization

# open("mpc.dat", "w") do io
#     serialize(io, mpc)
# end
#@time mpc = PowerFlow.rundcpf(mpc, opt);
# (2) Run the power flow from MATLAB
# mat"addpath('C:/Users/13733/Desktop/matpower7.1')"
# results=mat"solvedcase"
# vm_res = mpc["bus"][:, 8] - results["bus"][:, 8]
# va_res = mpc["bus"][:, 9] - results["bus"][:, 9]
# plot(vm_res, label="Voltage magnitude difference")
#  plot!(va_res, label="Voltage angle difference")
