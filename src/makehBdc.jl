function makehBdc(baseMVA, bus, branch)
    # constants
    (PQ, PV, REF,Pdc,Vdc,NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA,
    VM,VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN) = idx_bus();
    (F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C,TAP, SHIFT,
    BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST,ANGMIN, ANGMAX, MVSC1,
     MVSC2, BRANCHMODE, ETACR,ETACI, PHI, MU_ANGMIN, MU_ANGMAX) = idx_brch();
(GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, 
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, 
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF) = idx_gen();

    #initialize a zero matrix Bbus
    Bbus=zeros(size(bus,1),size(bus,1))
    #set Bbus = 1/R for each branch
    Bbus[CartesianIndex.(Int.(branch[:, F_BUS]), Int.(branch[:, T_BUS]))] .= 1.0./branch[:,BR_R]
    Bbus[CartesianIndex.(Int.(branch[:, T_BUS]), Int.(branch[:, F_BUS]))] .= 1.0./branch[:,BR_R]
    return  Bbus
end