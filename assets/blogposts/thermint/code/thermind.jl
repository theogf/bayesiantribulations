# This file was generated, do not modify it. # hide
using FastGaussQuadrature, LinearAlgebra
nodes, weights = gausshermite(100)
λs = range(0, 1, length= 100)
expec_λ = zeros(length(λs))
for (i, λ) in enumerate(λs)
    pow_post = power_posterior(y, λ)
    expec_λ[i] = dot(map(x->logpdf(likelihood(x), y), nodes * std(pow_post) .+ mean(pow_post)), weights)
end
plot(λs, expec_λ, label = L"E_{p_\lambda}[\log p(y|x)]", xlabel = L"\lambda")
savefig(joinpath(@OUTPUT, "thermint.svg")) # hide