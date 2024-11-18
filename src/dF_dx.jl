#this function is used to calculate the jacobian matrix of the power flow equations
function dF_dx(Gnm,Bnm,Gdc,Vn_ac,Vn_dc,θnm,Cc_ac,Cc_dc,Cg_ac)
    dP_dVn_ac1=-(Gnm.*cos.(θnm).+Bnm.*sin.(θnm)).*Vn_ac #dP_ac/dV_acm
    I_matrix = Matrix{Bool}(I, size(dP_dVn_ac1)...)
    dP_dVn_ac1_no_diag = dP_dVn_ac1 .* (1 .- I_matrix)
    dP_dVn_ac2=sum(2*Gnm.*Vn_ac,dims=2)-( Gnm.*cos.(θnm)+Bnm.*sin.(θnm))*Vn_ac #dP_ac/dV_acn
    diagP_dVn_ac=Diagonal(vec(dP_dVn_ac2))
    dP_dVn_ac=dP_dVn_ac1_no_diag+diagP_dVn_ac #dPac/dV_ac

    dQ_dVn_ac1=-(Gnm.*sin.(θnm).-Bnm.*cos.(θnm)).*Vn_ac #dQ_ac/dV_acm
    dQ_dVn_ac1_no_diag = dQ_dVn_ac1 .* (1 .- I_matrix)
    dQ_dVn_ac2=-sum(2*Bnm.*Vn_ac,dims=2)-( Gnm.*sin.(θnm)-Bnm.*cos.(θnm))*Vn_ac #dQ_ac/dV_acn
    diagQ_dVn_ac=Diagonal(vec(dQ_dVn_ac2))
    dQ_dVn_ac=dQ_dVn_ac1_no_diag+diagQ_dVn_ac #dQac/dV_ac

    dP_dθn_ac1=-(Gnm.*sin.(θnm).-Bnm.*cos.(θnm)).*Vn_ac.*Vn_ac'
    dP_dθn_ac1_no_diag = dP_dθn_ac1 .* (1 .- I_matrix)
    dP_dθn_ac2=-(-Gnm.*sin.(θnm).+Bnm.*cos.(θnm)).*Vn_ac*Vn_ac
    diagP_dθn_ac=Diagonal(vec(dP_dθn_ac2))
    dP_dθn_ac=dP_dθn_ac1_no_diag+diagP_dθn_ac #dPac/dθ_ac

    dQ_dθn_ac1=-(-Gnm.*cos.(θnm).-Bnm.*sin.(θnm)).*Vn_ac.*Vn_ac'
    dQ_dθn_ac1_no_diag = dQ_dθn_ac1 .* (1 .- I_matrix)
    dQ_dθn_ac2=-(Gnm.*cos.(θnm).+Bnm.*sin.(θnm)).*Vn_ac*Vn_ac
    diagQ_dθn_ac=Diagonal(vec(dQ_dθn_ac2))
    dQ_dθn_ac=dQ_dθn_ac1_no_diag+diagQ_dθn_ac #dQac/dθ_ac

    dP_dVn_dc1=-Gdc.*Vn_dc
    dP_dVn_dc2=sum((2Vn_dc.-Vn_dc').*Gdc,dims=2)
    diagP_dVn_dc=Diagonal(vec(dP_dVn_dc2))
    dP_dVn_dc=dP_dVn_dc1+diagP_dVn_dc #dPdc/dV_dc
    return dP_dVn_ac,dP_dVn_dc,dP_dθn_ac,dQ_dVn_ac,dQ_dθn_ac
end