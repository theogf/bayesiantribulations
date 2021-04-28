+++
title = "Gaussian Process Important derivations"
hasmath = true
hascode = true
blogpost = true
comment_section = true
+++
\newcommand{\KL}{\text{KL}}
\newcommand{\tr}{\text{tr}}
# Gaussian Process important derivations

\tableofcontents <!-- you can use \toc as well -->


$$
	\KL(q||p) = \frac{1}{2}\left[\log|K| - \log|\Sigma| - d + \tr (K^{-1}\Sigma) + (\mu_0 - \mu)^T K^{-1}(\mu_0 - \mu)\right].
$$



```julia:./code/kl
using KernelFunctions, LinearAlgebra, ForwardDiff, Plots, LaTeXStrings
using Random: seed! # hide
seed!(42) # hide
pyplot() # hide
default(guidefontsize=20.0, tickfontsize= 15.0) # hide
D = 3
μ = randn(D)
μ₀ = randn(D)
Σ = rand(D, D) |> x -> x * x' + I # Positive definite matrix
K = rand(D, D) |> x-> x * x' + I
L = cholesky(Σ).L # Cholesky of the covariance Σ
KL(μ, Σ::Matrix, μ₀ ,K) =
	0.5*(logdet(K) - logdet(Σ) - length(μ) +
		tr(inv(K) * Σ) + (μ₀ - μ)' * inv(K) * (μ₀ - μ))
@show KL(μ,Σ,μ₀,K)
```

\output{./code/kl}

When using instead $L$ where $LL^\top = \Sigma$.
\begin{align}
\KL(q||p) = \frac{1}{2}\left[\log|K| - 2 \log|L| - d + \text{tr} (L^\top K^{-1}L) + (\mu_0 - \mu)^T K^{-1}(\mu_0 - \mu)\right].
\end{align}

```julia:./code/kl
L = cholesky(Σ).L
KL(μ, L::LowerTriangular, μ₀, K) =
 	0.5*(logdet(K) - 2logdet(L) - length(μ) +
			tr(L' * inv(K) * L) + (μ₀ - μ)' * inv(K) * (μ₀-μ))
@show KL(μ,L,μ₀,K)
```
\output{./code/kl}

### Derivatives KL Divergence given variational parameters
#### Gradients
Gradients given $\mu,\Sigma,L$:
\begin{align}
\frac{d\KL}{d\mu} =& K^{-1}(\mu-\mu_0)\\
\frac{d\KL}{d\Sigma} =& \frac{1}{2}\left(-\Sigma^{-1} + K^{-1}\right)\\
\frac{d\KL}{dL} =& \frac{1}{2}\left(-2\text{diag}(L)^{-1} + 2K^{-1}L\right)
\end{align}
```julia:./code/kl
dKL_dμ(K, μ, μ₀) = inv(K) * (μ - μ₀) # Analytic Formulation
analytic = dKL_dμ(K, μ, μ₀)
autodiff = ForwardDiff.gradient(μ) do x
	KL(x, Σ, μ₀, K)
end
p = bar(1:D, abs.(analytic - autodiff), xlabel="μᵢ", ylabel=L"\Delta \frac{dKL}{d\mu_i}", label = "", xticks = 1:D) # hide
savefig(joinpath(@OUTPUT, "dKLdmu.svg")) # hide
```
\output{./code/kl}
\fig{./code/output/dKLdmu.svg}

```julia:./code/kl
dKL_dΣ(K, Σ) = 0.5 * (-inv(Σ) + inv(K))
analytic = dKL_dΣ(K, Σ)
autodiff = ForwardDiff.gradient(Σ) do x
	KL(μ, x, μ₀, K)
end
plot(heatmap(analytic, title="Analytic", colorbar = false), # hide
 		heatmap(autodiff, title="AutoDiff", colorbar = false), # hide
		heatmap(analytic - autodiff, title="Difference"), # hide
		yflip=true, layout=(1,3), ticks = 1:D, clims=extrema(vcat(analytic,autodiff))) # hide
savefig(joinpath(@OUTPUT,"dKLdSigma.svg")) # hide
```
\fig{./code/output/dKLdSigma.svg}

```julia:./code/kl
dKL_dL(K, L) =
	LowerTriangular(0.5 * (-2 * inv(Diagonal(Matrix(L))) + 2 * inv(K) * L))
analytic = dKL_dL(K, L)
autodiff = ForwardDiff.gradient(L) do x
	KL(μ, x, μ₀, K)
end
plot(heatmap(analytic,title="Analytic"), #hide
	heatmap(autodiff,title="AutoDiff"), #hide
	heatmap(analytic-autodiff,title="Difference"), #hide
	yflip=true,layout=(1,3), ticks = 1:D, clims=extrema(vcat(analytic,autodiff))) #hide
savefig(joinpath(@OUTPUT,"dKLdL.svg")) # hide
```
\fig{./code/output/dKLdL.svg}

#### Hessians
Hessian given $\mu,\Sigma,L$:
\begin{align}
\frac{d^2KL}{d\mu\mu} =& K^{-1}\\
\frac{d^2KL}{d\mu d\Sigma} =& 0\\
\frac{d^2KL}{d\Sigma\Sigma} =& \frac{1}{2} \Sigma^{-1}\otimes \Sigma^{-1}\\
\frac{d^2KL}{d\mu dL} =& 0\\
\frac{d^2KL}{dLL} =& - \text{diag} L^{-1}\otimes L^{-1} + K\otimes I
\end{align}
Where $\otimes$ is the outer-product\\

```julia:./code/kl
d2KL_dμμ(K) = inv(K)
analytic = d2KL_dμμ(K)
autodiff = ForwardDiff.hessian(μ) do x
	KL(x, Σ, μ₀, K)
end
plot(heatmap(analytic, title="Analytic"), # hide
		heatmap(autodiff, title="AutoDiff"), # hide
		heatmap(analytic - autodiff,title="Difference"), # hide
	yflip=true,layout=(D, 1), ticks = 1:D, clims=extrema(vcat(analytic,autodiff))) # hide
savefig(joinpath(@OUTPUT,"d2KLdmumu.svg")) # hide
```
\fig{./code/output/d2KLdmumu.svg}

```julia:./code/kl
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
```

\fig{./code/output/d2KLdSigmaSigma.svg}

<!--
### Derivatives KL Divergence given hyperparameters

Let $\theta$ be an hyperparameter (involved in computing $K$)

#### Gradients
\begin{align}
	\frac{d\KL}{d\theta_i} =& \frac{1}{2}\left[ \tr(K^{-1}J_i) + \tr\left(-K^{-1}J_iK^{-1}\underbrace{\left(\Sigma+(\mu_0-\mu)(\mu_0-\mu)^\top\right)}_{:=X}\right) \right]\\
	=& \frac{1}{2}\tr\left(K^{-1}J_i\left(I-K^{-1}X\right)\right)\\
	=& \frac{1}{2}\tr\left(J_i\underbrace{\left(I-K^{-1}X\right)K^{-1}}_{\text{precomputable}:=A}\right)
\end{align}

Where $J_i=\frac{dK}{d\theta_i}$ -->
