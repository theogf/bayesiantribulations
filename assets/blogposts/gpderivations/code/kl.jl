# This file was generated, do not modify it. # hide
d2KL_dSS(Σ) = 0.5*kron(inv(Σ),inv(Σ))
 analytic = d2KL_dSS(Σ)
# @show autodiff = ForwardDiff.hessian(Σ->KL(μ,Σ,μ₀,K),Σ)
# plot(heatmap(analytic,title="Analytic"),heatmap(autodiff,title="AutoDiff"),heatmap(analytic-autodiff,title="Difference"),yflip=true,layout=(1,3)) # hide
# savefig(joinpath(@OUTPUT,"d2KLdSigmaSigma.svg")) # hide