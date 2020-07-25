

@def title = "Thermodynamic Integration"
@def hasmath = true
@def comment_section = true
@def hacode = true

# Thermodynamic Integration

**Thermodynamic Integration (TI)** is a method coming straight from statistical physics to compute the difference of free energy between two systems $A$ and $B$. We are going to go from this physics definition and see how we can apply it to Bayesian Inference.

We start with a system with a potential energy function $U(x)$ with associated probability function

$$p(x) = \frac{1}{Z}\exp\left(-\frac{U(x)}{k_BT}\right)$$

Where $Z$  is the partition function of this system defined as

$$Z = \int_\Omega \exp\left(-\frac{U(x)}{k_BT}\right)dx$$

where $\Omega$ is the domain of all the states of the system and $x$ is one state of this system. The definition of the free energy is given by

$$F = -k_BT\log(Z)$$

Now going back to our systems $A$ and $B$ with potential energy $U_A$ and $U_B$ . We can construct a new system built as a linear interpolation between the two systems : ${U_\lambda = U_A + \lambda(U_B - U_A)}$

We can show that

$$F_B - F_A = \int_0^1 E_{U(\lambda)}\left[U_B - U_A\right]d\lambda$$

<details>
    <summary> Proof :</summary>
    $$\int_0^1 \frac{\partial F(\lambda)}{\partial \lambda}d\lambda = \int_0^1 \frac{\partial \log(Z(\lambda)}{\partial \lambda}d\lambda = $$
</details>



\begin{align}
F_B - F_A =& F(1) - F(0) = \int_0^1 \frac{\partial F(\lambda)}{\partial \lambda}d\lambda\\
\int_0^1 \frac{\partial F(\lambda)}{\partial \lambda}d\lambda &= -\int_0^1 \frac{\partial \log Z_\lambda}{\partial \lambda}d\lambda = -\int_0^1 \frac{1}{Z_\lambda}\frac{\partial Z_\lambda}{\partial \lambda}d\lambda \\
&=\int_0^1 \int_\Omega \frac{\exp(-U_\lambda(x))}{Z_\lambda}\frac{\partial U_\lambda(x)}{\partial \lambda}dxd\lambda\\
&=\int_0^1 E_{p_\lambda(x)}\left[\frac{dU\lambda(x)}{d\lambda}\right]d\lambda = \int_0^1 E_{p(x)}\left[U_B(x) - U_A(x)\right]d\lambda
\end{align}



It is all very nice, but very abstract if you are not a physicist! What does it have to do with Bayesian inference?

The Bayes theorem states that the posterior

$p(x|y) = \frac{p(x,y)}{p(y)}$ ,

where $x$ is my hypothesis or the parameters of my model, and $y$ is my observed data! We can actually rewrite as an energy model, as earlier!

$$p(x|y) = \frac{1}{Z}\exp\left(-U(x,y)\right)$$

where the potential energy $U(x,y) = -\log(p(x,y))$ is the negative log joint and the partition function $Z=\int_\Omega \exp(-U(x,y))dx$ is the evidence. We have furthermore the connection that the free energy is nothing else than the log evidence.

Now we have a direct connection between energy models and Bayesian problems! Which leads us to the use of TI for Bayesian inference:

The most difficult part of the Bayes theorem comes from the fact that except for simple cases, the posterior $p(x|y)$ is intractable. Most of the time the joint distribution $p(x,y)$ is known in closed form, the real issue is then to estimate the evidence $p(y)$, which is exactly what TI aims at.

The two systems we are going to consider is the prior : $U_A(x) = -\log p(x)$ and the joint distribution $U_B(x) = -\log p(x,y) = -\log p(y|x) - \log p(x)$. Now our intermediate state is given by

$$U_\lambda(x) = -\log p(x) + \lambda(-\log p(y|x) - \log p(x) - \log p(x)) = -\log p(x) - \lambda p(y|x)$$.

The normalized density derived from this potential energy is called a **power posterior** :

$$p_\lambda(x|y) = \frac{p(y|x)^\lambda p(y)}{Z_\lambda}$$

which just a posterior for which we reduced the importance of the likelihood.

Ok so performing the thermodynamic integration we derived earlier gives us :

$$ \log\int p(x,y)dx - \log \underbrace{\int p(x) dx}_{=1} = \log p(y) = \int_0^1 E_{p_\lambda(x)}[\log p(y|x)]d\lambda$$

Now let's put this to practice for a very simple use case a Gaussian prior with a Gaussian likelihood

\begin{align}
p(x) = \mathcal{N}(x|10, 1)
p(y|x) = \mathcal{N}(y|x, 1)
\end{align}

We take $y = -10$ to make a clear difference between $U_A$ and $U_B$

Of course the posterior can be found analytically but this will help us to evaluate different approaches.
```julia:./code/thermint
using Plots, Distributions, LaTeXStrings; pyplot(); default(lw = 2.0, legendfontsize = 15.0, labelfontsize = 15.0)
σ_prior = 1.0; σ_likelihood = 1.0
μ_prior = 10.0
prior = Normal(μ_prior, σ_prior)
likelihood(x) = Normal(x, σ_likelihood)
y = -10.0
p_prior(x) = pdf(prior, x)
p_likelihood(x, y) = pdf(likelihood(x), y)
```
\output{./code/thermint}

Now we have all the parameters in place we can define the exact posterior and power posterior:

```julia:./code/thermint
σ_posterior(λ) = sqrt(inv(1/σ_prior^2 + λ / σ_likelihood^2))
μ_posterior(y, λ) = σ_posterior(λ) * (λ * y / σ_likelihood^2 + μ_prior / σ_prior^2)
posterior(y) = Normal(μ_posterior(y, 1.0), σ_posterior(1.0))
p_posterior(x, y) = pdf(posterior(y), x)
power_posterior(y, λ) = Normal(μ_posterior(y, λ), σ_posterior(λ))
p_pow_posterior(x, y, λ) = pdf(power_posterior(y, λ), x)
xgrid = range(-10, 10, length = 400)
plot(xgrid, p_prior, label = "Prior", xlabel = "x")
plot!(xgrid, x->p_likelihood(x, y),label =  "Likelihood")
plot!(xgrid, x->p_posterior(x, y), label = "Posterior")
savefig(joinpath(@OUTPUT, "distributions.svg")) # hide
plot(xgrid, [x->p_pow_posterior(x, y, λ) for λ in 0:0.2:1], label = reshape(["λ = $λ" for λ in 0:0.2:1], 1, :), title = "Power Posteriors", xlabel = "x", ylabel = L"p_\lambda(x|y)") # hide
savefig(joinpath(@OUTPUT, "power_posteriors.svg")) # hide
```

\output{./code/thermint}
\fig{./code/output/distributions.svg}
\fig{./code/output/power_posteriors.svg}

We can also visualize the energies themselves
```julia:./code/thermint
U_A(x) = -logpdf(prior, x)
U_B(x, y) = -logpdf(likelihood(x), y) - logpdf(prior, x)
U_λ(x, y, λ) = -logpdf(prior, x) - λ * logpdf(likelihood(x), y)
plot(xgrid, [x->U_λ(x, y, λ) for λ in 0:0.2:1], label = reshape(["λ = $λ" for λ in 0:0.2:1], 1, :), title = "Potential Energy") # hide
plot!(xgrid, U_A, line = :dash, label = L"U_A", xlabel = "x", ylabel = L"U(x)") # hide
plot!(xgrid, x->U_B(x, y), line = :dash, label = L"U_B")  # hide
savefig(joinpath(@OUTPUT, "energies.svg")) # hide
```
\output{./code/thermint}
\fig{./code/output/energies.svg}

Now we can start evaluating the integrand for multiple $\lambda$ :

```julia:./code/thermint
M = 100
λs = range(0, 1, length= M)
λs = ((1:M)./M).^5
expec_λ = zeros(length(λs))
T = 1000
for (i, λ) in enumerate(λs)
    pow_post = power_posterior(y, λ)
    expec_λ[i] = sum(logpdf(likelihood(x), y) for x in rand(pow_post, T)) / T
end
plot(λs, expec_λ, label = L"E_{p_\lambda}[\log (p(y|x))]", xlabel = L"\lambda") # hide
savefig(joinpath(@OUTPUT, "expec_log.svg")) # hide
```
\output{./code/thermint}
\fig{./code/output/expec_log.svg}

And we can now compare the sum with the actual value of $Z$ :

```julia:./code/thermint
using Trapz
logpy = logpdf(posterior(y), y)
TI_logpy = [trapz(λs[1:i], expec_λ[1:i]) for i in 1:length(λs)]
plot(λs, TI_logpy, label = "Therm. Int.", xlabel = L"\lambda") # hide
hline!([logpy], label = latexstring("\\log (p(y)) = ", logpy)) # hide
savefig(joinpath(@OUTPUT, "thermint.svg")) # hide
```
\output{./code/thermint}
\fig{./code/output/thermint.svg}

Now that's great but how does it compare to other methods?
Well we want to compute the integral $p(y) = \int p(y|x)p(x)dx$. The most intuitive way is to sample from the prior $p(x)$ to perform a Monte-Carlo integration:

$$\int p(y|x)p(x)dx \approx \frac{1}{N}\sum_{i=1}^N p(y|x_i)$$

where $x_i \sim p(x)$.

```julia:./code/thermint
T = 10000
xs = rand(prior, T)
prior_logpy = [log(mean(pdf.(likelihood.(xs[1:i]), y))) for i in 1:T]
plot(1:T, prior_logpy, label = "Prior MC Integration", xlabel = "T") # hide
hline!([logpy], label = latexstring("\\log (p(y)) = ", logpy)) # hide
savefig(joinpath(@OUTPUT, "prior_integration.svg")) # hide
```

\output{./code/thermint}
\fig{./code/output/prior_integration.svg}

As you can see the result is quite off. The reason is that it happens a lot that the prior and the likelihood have very little overlap. This leads to a huge variance in the estimate of the integral.

Finally there is another approach called the **harmonic mean estimator**. So far we sampled from *power posteriors*, *the prior* but not from the posterior!
Most Bayesian inference methods aim at getting samples from the posterior so this would look like an appropriate approach.
If we perform naive **importance sampling** :
$$\int \frac{p(x,y)}{p(x|y)}p(x|y)dx \approx \frac{1}{N}\frac{p(x_i,y)}{p(x_i|y)} = p(y),$$
this would simply not work.
But if we use an unnormalized version of the density $\tilde{p}(x|y) \propto p(x|y)$ we get :
$$ p(y) = \frac{\int \frac{p(y|x)p(x)}{\tilde{p}(x|y)}p(x|y)dx}{\int \frac{p(x)}{\tilde{p}(x|y)}p(x|y)dx} $$
Now replacing $\tilde{p}(x|y)$ by $p(y|x)p(x)$ we get the equation :
$$ p(y) = \frac{1}{E_{p(x|y)}\left[\frac{1}{p(y|x)}\right]}\approx \left[\frac{1}{N}\sum \frac{1}{p(y|x_i)}\right]^{-1} $$
where $x_i \sim p(x|y)$.
Now the issue with this is that even though both side of the fractions are unbiased, the ratio is not! Experiments tend show this leads to a large bias. Let's have a look for our problem

```julia:./code/thermint
T = 10000
xs = rand(posterior(y), T)
posterior_logpy = [-log(mean(inv.(pdf.(likelihood.(xs[1:i]), y)))) for i in 1:T]
plot(1:T, posterior_logpy, label = "Posterior MC Integration", xlabel = "T") # hide
hline!([logpy], label = latexstring("\\log (p(y)) = ", logpy)) # hide
savefig(joinpath(@OUTPUT, "posterior_integration.svg")) # hide
```
\output{./code/thermint}
\fig{./code/output/posterior_integration.svg}
