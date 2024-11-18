function hybridnewtonpf(Ybus,Bbus, ref, pv, pq,pdc,vdc,bus_ac,bus_dc,converter,gen_ac,baseMVA)
    #the content is belong to hybridnewtonpf
 (GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, 
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, 
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF) = idx_gen();
Gnm=real.(Ybus)
Bnm=imag.(Ybus)
Gdc=Bbus
on = findall(gen_ac[:, GEN_STATUS] .> 0)

Cc_ac=zeros(size(bus_ac,1),size(converter,1))
for i in 1:size(converter,1)
    Cc_ac[Int.(bus_dc[i,2]),i]=1
end
Cc_dc=zeros(size(bus_dc,1),size(converter,1))
for i in 1:size(converter,1)
    Cc_dc[Int.(bus_dc[i,1]),i]=1
end
Cg_ac=zeros(size(bus_ac,1),size(bus_ac,1))
for i in 1:size(on,1)
    Cg_ac[Int.(on[i]),i]=1
end
# Pc_ac=-converter[:,4]./baseMVA
# Qc_ac=-converter[:,5]./baseMVA
# Pc_dc=zeros(size(converter,1),1)
Pg_ac=zeros(size(bus_ac,1),1)
for i in 1:size(gen_ac,1)
    Pg_ac[Int.(gen_ac[i,1]),1]=gen_ac[i,2]./baseMVA
end
Qg_ac=zeros(size(bus_ac,1),1)
for i in 1:size(gen_ac,1)
    Qg_ac[Int.(gen_ac[i,1]),1]=gen_ac[i,3]./baseMVA
end
ηi=converter[:,20]
ηr=converter[:,19]
bus_ac,bus_dc=runmodel(bus_ac,bus_dc,converter,Gnm,Bnm,Gdc,Cc_ac,Cc_dc,Pg_ac,Qg_ac,Cg_ac,ref,pv,pq,pdc,vdc,ηi,ηr,baseMVA)
return bus_ac,bus_dc
#Evaluate F(x0)
# Pcal_ac=sum(Gnm.*Vn_ac.^2,dims=2).-(Gnm.*cos.(θnm) .+Bnm.*sin.(θnm)).*Vn_ac*Vn_ac+Cc_ac*Pc_ac-Cc_ac*(Pc_dc.*ηi)
# Qcal_ac=-sum(Bnm.*Vn_ac.^2,dims=2).-(Gnm.*sin.(θnm) .-Bnm.*cos.(θnm)).*Vn_ac*Vn_ac+Cc_ac*Qc_ac
# Pcal_dc=Gdc*( Vn_dc'.-Vn_dc)*Vn_dc+Cc_dc*Pc_dc-Cc_dc*(Pc_ac.*ηr)
# #Calculate mis 
# mis_ac=Pcal_ac+im.*Qcal_ac-Cg_ac*Pg_ac+Pd_ac-im.*(Cg_ac*Qg_ac-Qd_ac)
# mis_dc=Pcal_dc-Pd_dc
# F=[real(mis_ac[vcat(pq,pv)]);imag(mis_ac[pq]);mis_dc[pdc]]
# #check tolerance
# normF = PowerFlow.norm(F, Inf)
# if normF < tol
#     converged = true
# end
# #Do newton iteration
# while (!converged && i < max_it)

#     #update iteration counter
#     i+=1
#     #evaluate Jacobian
#     #calculate the jacobi element of ACDC part
#     dP_dVn_ac,dP_dVn_dc,dP_dθn_ac,dQ_dVn_ac,dQ_dθn_ac=dF_dx(Gnm,Bnm,Gdc,Vn_ac,Vn_dc,θnm,Cc_ac,Cc_dc,Cg_ac)

# end

end