"""
    Convert the data from external format to internal format
"""
function ext2int(bus::Matrix{Float64}, gen::Matrix{Float64}, branch::Matrix{Float64})
    (PQ, PV, REF,Pdc,Vdc,NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA,
    VM,VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN) = idx_bus();
      (F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C,TAP, SHIFT,
       BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST,ANGMIN, ANGMAX, MVSC1,
        MVSC2, BRANCHMODE, ETACR,ETACI, PHI, MU_ANGMIN, MU_ANGMAX) = idx_brch();
    (GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, 
        MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, 
        QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF) = idx_gen();
    #TODO: determine which buses, branches, gens are connected & in-service
    gen = gen[gen[:, 8] .!= 0, :]
    branch = branch[branch[:, 11] .!= 0, :]
    # create map of external bus numbers to bus indices
    i2e = bus[:, BUS_I]
    e2i = sparsevec(zeros(Int, Int(maximum(i2e))))
    e2i[Int.(i2e)] = 1:size(bus, 1)
    # renumber buses consecutively
    bus[:, BUS_I] = e2i[bus[:, BUS_I]]
    gen[:, GEN_BUS] = e2i[gen[:, GEN_BUS]]
    branch[:, F_BUS] = e2i[branch[:, F_BUS]]
    branch[:, T_BUS] = e2i[branch[:, T_BUS]]
    return bus, gen, branch
end
