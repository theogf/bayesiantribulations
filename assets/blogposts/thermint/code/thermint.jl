# This file was generated, do not modify it. # hide
T = 10000
xs = rand(posterior(y), T)
posterior_logpy = [-log(mean(inv.(pdf.(likelihood.(xs[1:i]), y)))) for i in 1:T]
plot(1:T, posterior_logpy, label = "Posterior MC Integration", xlabel = "T") # hide
hline!([logpy], label = latexstring("\\log (p(y)) = ", logpy)) # hide
savefig(joinpath(@OUTPUT, "posterior_integration.svg")) # hide