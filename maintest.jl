"""
    Main function for the AC power flow
"""

# Detect the current working operating system
if Sys.iswindows()
    # Add the path to the src folder
    include(joinpath(@__DIR__, "src", "PowerFlow.jl"))
    include(joinpath(@__DIR__, "data", "case7h.jl"))
else
    # Add the path to the src folder
    include(joinpath(@__DIR__, "src", "PowerFlow.jl"))
    include(joinpath(@__DIR__, "data", "case118.jl"))
    using AppleAccelerate
end
# push!(LOAD_PATH, pwd()*"\\data\\");
using .PowerFlow
# using MATLAB
# using Plots
# Start a MATLAB engine
# mat"addpath('C:\\Codes\\matpower')"
# mpc = case118();
#opt = PowerFlow.options() # The initial settings 
# opt["PF"]["NR_ALG"] = "bicgstab";
# opt["PF"]["ENFORCE_Q_LIMS"]=0
mpc=case7h()
#run power flow
 @time mpc = PowerFlow.run_hybridpf(mpc)
 println(mpc["busAC"])
 println(mpc["busDC"])