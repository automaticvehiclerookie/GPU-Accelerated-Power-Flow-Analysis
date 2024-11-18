"""
   Main function to call the DC power flow function
    Input: case file
    Output: results of the power flow as a dictionary
    Example:
    bus, gen, branch = rundcpf(casefile)
"""
function rundcpf(mpc, opt::Dict{String})
   opt["PF"]["DC"] = 1
    runpf(mpc, opt)
end