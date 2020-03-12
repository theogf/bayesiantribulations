@def title = "Gaussian Process Important derivations"
@def hasmath = true
@def hascode = true

# Gaussian Process important derivations

\tableofcontents <!-- you can use \toc as well -->


## KL Divergence for Multivariate Gaussians
\begin{align}
	\KL(q||p) = \frac{1}{2}\left[\log|K| - \log|\Sigma| - d + \text{tr} (K^{-1}\Sigma) + (\mu_0 - \mu)^T K^{-1}(\mu_0 - \mu)\right].
\end{align}

```julia:./code/kl
using KernelFunctions, LinearAlgebra, ForwardDiff, Plots, LaTeXStrings
pyplot(); default(guidefontsize=15.0)#hide
μ = randn(3)
μ₀ = randn(3)
Σ = rand(3,3) |> x->x*x' # Positive definite matrix
K = rand(3,3) |> x->x*x'
L = cholesky(Σ).L
KL(μ,Σ::Matrix,μ₀,K) = 0.5*(logdet(K)-logdet(Σ)-length(μ)+tr(inv(K)*Σ)+(μ₀-μ)'*inv(K)*(μ₀-μ))
@show KL(μ,Σ,μ₀,K)
```
\output{./code/kl}

When using instead $L$ where $LL^\top = \Sigma$.
\begin{align}
\KL(q||p) = \frac{1}{2}\left[\log|K| - 2 \log|L| - d + \text{tr} (L^\top K^{-1}L) + (\mu_0 - \mu)^T K^{-1}(\mu_0 - \mu)\right].
\end{align}

```julia:./code/kl
L = cholesky(Σ).L
KL(μ,L::LowerTriangular,μ₀,K) = 0.5*(logdet(K)-2*logdet(L)-length(μ)+tr(L'*inv(K)*L)+(μ₀-μ)'*inv(K)*(μ₀-μ))
@show KL2(μ,L,μ₀,K)
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
dKL_dμ(K,μ,μ₀) = inv(K)*(μ-μ₀)
bar(abs.(dKL_dμ(K,μ,μ₀)-ForwardDiff.gradient(μ->KL(μ,Σ,μ₀,K),μ)),xlabel="i",ylabel=L"\Delta \frac{dKL}{d\mu_i}";) # hide
savefig(joinpath(@OUTPUT,"dKLdmu.svg")) # hide
```
\fig{./code/kl/output/dKLdmu.svg}
```julia:./code/kl
dKL_dΣ(K,Σ) = 0.5*(-inv(Σ)+inv(K))
plot(heatmap(dKL_dΣ(K,Σ),title="Analytic"),heatmap(ForwardDiff.gradient(Σ->KL(μ,Σ,μ₀,K),Σ),title="AutoDiff"),heatmap(dKL_dΣ(K,Σ)-ForwardDiff.gradient(Σ->KL(μ,Σ,μ₀,K),Σ),title="Difference"),yflip=true,layout=(1,3)) # hide
savefig(joinpath(@OUTPUT,"dKLdSigma.svg")) # hide
```
\fig{./code/kl/output/dKLdSigma.svg}

```julia:./code/kl
dKL_dL(K,L) = LowerTriangular(0.5*(-2*inv(Diagonal(Matrix(L)))+2*inv(K)*L))
analytic = dKL_dL(K,L) #hide
autodiff = ForwardDiff.gradient(L->KL(μ,L,μ₀,K),L) # hide
plot(heatmap(analytic,title="Analytic"),heatmap(autodiff,title="AutoDiff"),heatmap(analytic-autodiff,title="Difference"),yflip=true,layout=(1,3),clims=extrema(vcat(analytic,autodiff))) #hide
savefig(joinpath(@OUTPUT,"dKLdL.svg")) # hide
```
\fig{./code/kl/output/dKLdL.svg}
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
analytic = d2KL_dμμ(K) # hide
autodiff = ForwardDiff.hessian(μ->KL(μ,Σ,μ₀,K),μ) # hide
 plot(heatmap(analytic,title="Analytic"),heatmap(autodiff,title="AutoDiff"),heatmap(analytic-autodiff,title="Difference"),yflip=true,layout=(1,3)) # hide
savefig(joinpath(@OUTPUT,"d2KLdmumu.svg")) # hide
```
\fig{./code/kl/output/d2KLdmumu.svg}

```julia:./code/kl
 d2KL_dSS(Σ) = 0.5*kron(inv(Σ),inv(Σ))
 analytic = d2KL_dSS(Σ)
# @show autodiff = ForwardDiff.hessian(Σ->KL(μ,Σ,μ₀,K),Σ)
# plot(heatmap(analytic,title="Analytic"),heatmap(autodiff,title="AutoDiff"),heatmap(analytic-autodiff,title="Difference"),yflip=true,layout=(1,3)) # hide
# savefig(joinpath(@OUTPUT,"d2KLdSigmaSigma.svg")) # hide
```

<!--
<!--\fig{./code/derivgp/output/d2KLdmumu.svg}-->


### Derivatives KL Divergence given hyperparameters

Let $\theta$ be an hyperparameter (involved in computing $K$)

#### Gradients
\begin{align}
	\frac{d\KL}{d\theta_i} =& \frac{1}{2}\left[ \tr(K^{-1}J_i) + \tr\left(-K^{-1}J_iK^{-1}\underbrace{\left(\Sigma+(\mu_0-\mu)(\mu_0-\mu)^\top\right)}_{:=X}\right) \right]\\
	=& \frac{1}{2}\tr\left(K^{-1}J_i\left(I-K^{-1}X\right)\right)\\
	=& \frac{1}{2}\tr\left(J_i\underbrace{\left(I-K^{-1}X\right)K^{-1}}_{\text{precomputable}:=A}\right)
\end{align}

Where $J_i=\frac{dK}{d\theta_i}$
