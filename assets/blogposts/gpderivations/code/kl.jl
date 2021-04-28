# This file was generated, do not modify it. # hide
d2KL_dSS(Σ) = 0.5 * kron(inv(Σ), inv(Σ))
analytic = d2KL_dSS(Σ)
autodiff = ForwardDiff.hessian(Σ) do x
	KL(μ, x, μ₀, K)
end
plot(heatmap(analytic,title="Analytic"), # hide
		heatmap(autodiff,title="AutoDiff"), # hide
		heatmap(analytic-autodiff,title="Difference"), # hide
		yflip=true, layout=(D, 1), ticks = 1:D^2, clims=extrema(vcat(analytic,autodiff))) # hide
savefig(joinpath(@OUTPUT,"d2KLdSigmaSigma.svg")) # hide