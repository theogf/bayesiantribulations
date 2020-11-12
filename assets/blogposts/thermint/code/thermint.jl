# This file was generated, do not modify it. # hide
bar(["log p(y)", "TI", "Prior", "Posterior"], [logpy, TI_logpy[end], prior_logpy[end], posterior_logpy[end]], lab="", ylabel="log Z") # hide
savefig(joinpath(@OUTPUT, "comparison_methods.svg")) # hide