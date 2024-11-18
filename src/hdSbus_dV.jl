#the function is used to calculate the dF/dx in hybrid AC/DC grid
function hdSbus_dV()
    # Initialize Unm as a zero matrix with the same size as the number of buses
    Unm=zeros(size(bus, 1), size(bus, 1))

    # Set Unm[n, m] = 1 if a line connects buses n and m
    Unm[CartesianIndex.(Int.(branch[:, F_BUS]), Int.(branch[:, T_BUS]))] .= 1
    Unm[CartesianIndex.(Int.(branch[:, T_BUS]), Int.(branch[:, F_BUS]))] .= 1
    
    W=zeros(size(bus,1),1)
    W=bus[:,BUS_TYPE] .> 3

    # Initialize D as a ones matrix with the same size as the number of buses
    D = zeros(size(bus, 1), size(bus, 1))

    # Set D[n, m] = 0 if the branch connecting buses n and m is in AC mode and the bus types of n and m are different
    D[CartesianIndex.(Int.(branch[:, F_BUS]), Int.(branch[:, T_BUS]))]=branch[:,BRANCHMODE]
    D[CartesianIndex.(Int.(branch[:, T_BUS]), Int.(branch[:, F_BUS]))]=branch[:,BRANCHMODE]

    #obtain the phi from branch
    phi=zeros(size(bus,1),size(bus,1))
    phi[CartesianIndex.(Int.(branch[:, F_BUS]), Int.(branch[:, T_BUS]))]=branch[:,PHI].* pi / 180
    phi[CartesianIndex.(Int.(branch[:, T_BUS]), Int.(branch[:, F_BUS]))]=branch[:,PHI].* pi / 180

    theta_diff = (theta .- theta') .* pi / 180
    Ybus_real = real.(Ybus)
    Ybus_imag = imag.(Ybus)

    #calculate the H dP/dVa
    H=zeros(size(bus, 1), size(bus, 1))
    H=Unm.*(1 .- W).*(1 .- W').*(1 .- D).*Vm.*Vm'.*(Ybus_real.*sin.(theta_diff).-Ybus_imag.*cos.(theta_diff))
    #calculate the diagonal element
    H2=zeros(size(bus, 1), size(bus, 1))
    H2 = -Unm .* (1 .- W) .* (1 .- W') .* (1 .- D) .* (-Vm .* Vm' .* (-Ybus_real .* cos.(theta_diff) .+ Ybus_imag .* sin.(theta_diff)))
    H2=sum(H2, dims=2)
    for i=1:size(H2,1)
        H[i,i]=H2[i]
    end

    #calculate the N dP/dVm Vm
    N=zeros(size(bus, 1), size(bus, 1))
    term1=-Vm.*(Ybus_real.*cos.(theta_diff).+Ybus_imag.*sin.(theta_diff))
    term2=Gdc.*(-M.^-1 .*Vm.*M'.^-1).*(0.5 .* (1 .+ sign.(M .^ -1 .* Vm .- (M' .^ -1) .* Vm'))./etacr .+  0.5 .* (1 .- sign.(M .^ -1 .* Vm .- (M' .^ -1) .* Vm')).*etaci)
    term3=Gdc.*(-M.^-1 .*Vm).*(0.5 .* (1 .+ sign.(M .^ -1 .* Vm .- Vm'))./etacr.+  0.5 .* (1 .- sign.(M .^ -1 .* Vm .- Vm')).*etaci)
    term4=Gdc.*(-Vm.*M'.^-1)
    term5=Gdc.*(-Vm)
    N=  -Unm.*(1 .- W).*(1 .- W').*(1 .-D).*term1.*Vm'
    N+= -Unm.*(1 .- W).*(2 .-W').*D.*term2.*Vm'
    N+= -Unm.*(1 .-W).* W'.*D.*term3.*Vm'
    N+= -Unm.*W.*(1 .-W').*D.*term4.*Vm'
    N+= -Unm.*W.*W'.*D.*term5.*Vm'
    #calculate the diagonal element
    N2=zeros(size(bus, 1), size(bus, 1))
    term1= 2 .*Vm .* Ybus_real .- Vm'.*(Ybus_real.*cos.(theta_diff).+Ybus_imag.*sin.(theta_diff))
    term2=Gdc .* (2 .*M.^-2 .*Vm .-M.^-1 .*M'.^-1 .*Vm').*(0.5 .* (1 .+ sign.(M .^ -1 .* Vm .- (M' .^ -1) .* Vm'))./etacr .+  0.5 .* (1 .- sign.(M .^ -1 .* Vm .- (M' .^ -1) .* Vm')).*etaci)
    term3=Gdc .* (2 .*M.^-2 .*Vm .-M.^-1 .*Vm').*(0.5 .* (1 .+ sign.(M .^ -1 .* Vm .- Vm'))./etacr.+  0.5 .* (1 .- sign.(M .^ -1 .* Vm .- Vm')).*etaci)
    term4=Gdc .*(2 .*Vm .-M'.^-1 .*Vm')
    term5=Gdc .*(2 .*Vm .-Vm')
    N2 = -Unm.*(1 .- W).*(1 .- W').*(1 .-D).*term1.*Vm
    N2+= -Unm.*(1 .- W).*(1 .-W').*D.*term2.*Vm
    N2+= -Unm.*(1 .-W).* W'.*D.*term3.*Vm
    N2+= -Unm.*W.*(1 .-W').*D.*term4.*Vm
    N2+= -Unm.*W.*W'.*D.*term5.*Vm
    N2=sum(N2, dims=2)
    for i=1:size(N2,1)
        N[i,i]=N2[i]
    end
    
    #calculate the J dQ/dVa
    J=zeros(size(bus, 1), size(bus, 1))
    J=-Unm.*(1 .- W).*(1 .- W').*(1 .- D).*Vm.*Vm'.*(Ybus_real.*cos.(theta_diff).+Ybus_imag.*sin.(theta_diff))
    #calculate the diagonal element
    J2=zeros(size(bus, 1), size(bus, 1))
    J2 = -Unm .* (1 .- W) .* (1 .- W') .* (1 .- D) .* (-Vm .* Vm' .* (Ybus_real .* sin.(theta_diff) .+ Ybus_imag .* cos.(theta_diff)))
    J2=sum(J2, dims=2)
    for i=1:size(J2,1)
        J[i,i]=J2[i]
    end

    #calculate the L dQ/dVm Vm
    L=zeros(size(bus, 1), size(bus, 1))
    term1=-Vm.*(Ybus_real.*sin.(theta_diff).-Ybus_imag.*cos.(theta_diff))
    term2=Gdc.*(-M.^-1 .*Vm.*M'.^-1).*(0.5 .* (1 .+ sign.(M .^ -1 .* Vm .- (M' .^ -1) .* Vm'))./etacr .+  0.5 .* (1 .- sign.(M .^ -1 .* Vm .- (M' .^ -1) .* Vm')).*etaci)
    term3=Gdc.*(-M.^-1 .*Vm).*(0.5 .* (1 .+ sign.(M .^ -1 .* Vm .- Vm'))./etacr.+  0.5 .* (1 .- sign.(M .^ -1 .* Vm .- Vm')).*etaci)
    L=  -Unm.*(1 .- W).*(1 .- W').*(1 .-D).*term1.*Vm'
    L+= -Unm.*(1 .- W).*(1 .-W').*D.*term2.*Vm'
    L+= -Unm.*(1 .-W).* W'.*D.*term3.*Vm'
    #calculate the diagonal element
    L2=zeros(size(bus, 1), size(bus, 1))
    term1= -2 .*Vm .* Ybus_imag .- Vm'.*(Ybus_real.*sin.(theta_diff).+Ybus_imag.*cos.(theta_diff))
    term2=Gdc .* (2 .*M.^-2 .*Vm .-M.^-1 .*M'.^-1 .*Vm').*(0.5 .* (1 .+ sign.(M .^ -1 .* Vm .- (M' .^ -1) .* Vm'))./etacr .+  0.5 .* (1 .- sign.(M .^ -1 .* Vm .- (M' .^ -1) .* Vm')).*etaci).*tan(phi)
    term3=Gdc .* (2 .*M.^-2 .*Vm .-M.^-1 .*Vm').*(0.5 .* (1 .+ sign.(M .^ -1 .* Vm .- Vm'))./etacr.+  0.5 .* (1 .- sign.(M .^ -1 .* Vm .- Vm')).*etaci).*tan(phi)
    L2 = -Unm.*(1 .- W).*(1 .- W').*(1 .-D).*term1.*Vm
    L2+= -Unm.*(1 .- W).*(1 .-W').*D.*term2.*Vm
    L2+= -Unm.*(1 .-W).* W'.*D.*term3.*Vm
    L2=sum(L2, dims=2)
    for i=1:size(L2,1)
        L[i,i]=L2[i]
    end
end