# This file was generated, do not modify it. # hide
using Distributions
nu = 2.0
studentt = TDist(nu)
pomega = InverseGamma(nu, nu)
plot(-5:0.01:5, x->pdf(studentt, x), label="Student-T")

plot!()