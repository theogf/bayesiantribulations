# This file was generated, do not modify it. # hide
using Trapz
logZ = logpdf(posterior(y), y)
# TI_logZ = logpdf(prior, mean(prior)) .- cumsum(expec_λ[1:end-1]) * step(λs) # hide
TI_logZ = logpdf(prior, mean(prior)) .- [trapz(expec_λ[1:i], λs[1:i]) for i in 1:length(λs)]
plot(λs, TI_logZ, label = "Therm. Int.", xlabel = L"\lambda")
hline!([logZ], label = "logZ = $logZ")
savefig(joinpath(@OUTPUT, "thermint.svg")) # hide