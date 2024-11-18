"""
   Main function to call the AC/DC hybrid power flow function
    Input: case file
    Output: results of the power flow as a dictionary
    Example:
    bus, gen, branch = run_hybridpf(casefile)
"""
function run_hybridpf(mpc::Dict)
    (PQ, PV, REF,Pdc,Vdc,NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA,
 VM,VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN) =idx_bus();
 (F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C,TAP, SHIFT,
 BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST,ANGMIN, ANGMAX, MVSC1,
  MVSC2, BRANCHMODE, ETACR,ETACI, PHI, MU_ANGMIN, MU_ANGMAX) = idx_brch();
(GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, 
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, 
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF) = idx_gen();

baseMVA = mpc["baseMVA"];
bus_ac = mpc["busAC"];
bus_dc = mpc["busDC"];
gen_ac = mpc["genAC"];
branch_ac = mpc["branchAC"];
branch_dc = mpc["branchDC"];
converter = mpc["converter"];
(bus_ac, gen_ac, branch_ac) = ext2int(bus_ac, gen_ac, branch_ac);
#get the bustype from the bus matrix
(ref, pv, pq) = bustypes(bus_ac, gen_ac);
#TODO:rectify the bus type function
(pdc,vdc) =hybrid_bustypes(bus_dc,converter);

##-----  run the power flow  ----- 

#make Ybus
#TODO: 交流网连通，平衡节点位于交流网络中
Ybus, Yf, Yt = makeYbus(baseMVA, bus_ac, branch_ac);
#make Bbus
#TODO:
Bbus = makehBdc(baseMVA, bus_dc, branch_dc);
#Sbus = Vm -> PowerFlow.makeSbus(baseMVA, bus, gen, Vm);
#newton power flow
bus_ac,bus_dc= hybridnewtonpf(Ybus,Bbus, ref, pv, pq,pdc,vdc,bus_ac,bus_dc,converter,gen_ac,baseMVA)
#return the results
mpc["busAC"] = bus_ac;
mpc["busDC"] = bus_dc;
return mpc
end