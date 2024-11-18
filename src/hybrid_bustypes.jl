function hybrid_bustypes(bus::Matrix{Float64},converter::Matrix{Float64})
    # constants
    (PQ, PV, REF,Pdc,Vdc,NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA,
    VM,VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN) = idx_bus();
    (GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, 
        MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, 
        QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF) = idx_gen();
    # get the bustype from the bus matrix
    pdc=findall(converter[:, BUS_TYPE] .== 1)
    vdc=findall(converter[:, BUS_TYPE] .== 2)

    return pdc,vdc
end