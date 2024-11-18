# Laplacian 2D
const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots

@parallel function compute_1!(dAdt, A, D, dt, dmp, dx, dy)
    @inn(dAdt) = @inn(dAdt)*(1.0-dmp) + dt*D*( @d2_xi(A)/(dx*dx) + @d2_yi(A)/(dy*dy) )
    return
end

@parallel function compute_2!(A, dAdt, dt)
    @all(A) = @all(A) + dt*@all(dAdt)
    return
end

@views function laplacian2D()
    fact    = 40
    # Physics
    lx, ly  = 10, 10
    D       = 1
    # Numerics
    nx, ny  = fact*50, fact*50
    dx, dy  = lx/nx, ly/ny
    niter   = 20*nx
    dmp     = 2.0/nx
    dt      = dx/sqrt(D)/2.1
    # Initial conditions
    A       = @zeros(nx, ny)
    dAdt    = @zeros(nx, ny)
    A[2:end-1,2:end-1] .= @rand(nx-2, ny-2)
    # display(heatmap(Array(A)', aspect_ratio=1, xlims=(1,nx), ylims=(1,ny))); error("initial condition")
    errv = []
    # iteration loop
    for it = 1:niter
        @parallel compute_1!(dAdt, A, D, dt, dmp, dx, dy)
        @parallel compute_2!(A, dAdt, dt)
        if it % nx == 0
            err = maximum(abs.(A)); push!(errv, err)
            p1=plot(nx:nx:it,log10.(errv), linewidth=3, markersize=4, markershape=:circle, framestyle=:box, legend=false, xlabel="iter", ylabel="log10(max(|A|))", title="iter=$it")
            p2=heatmap(Array(A)', aspect_ratio=1, xlims=(1,nx), ylims=(1,ny), title="max(|A|)=$(round(err,sigdigits=3))")
            display(plot(p1,p2, dpi=150))
        end
    end
    return
end

@time laplacian2D()