@def title = "Thermodynamic Integration"
@def hasmath = true
@def comment_section = true
@def blog_post = true
@def hascode = true

# Thermodynamic Integration

**Thermodynamic Integration (TI)** is a method coming straight from statistical physics.
In physics, it is used to compute the difference of free energy between two systems $A$ and $B$ by creating a path between the two.
This might sound sounds very abstract but we are going to go from explaining what it means in physics to a direct application in Bayesian Inference.

## The physics approach

Let's first define a simple system, with its state defined by $x$.
For example, we could pick a spring (the system), without gravity, friction, etc, and define $x$ as the distance from the rest position.

![Simple illustration of the potential energy of a spring](/assets/thermint/spring.gif)
Illustration from [http://hydrogen.physik.uni-wuppertal.de/hyperphysics/hyperphysics/hbase/pespr.html](http://hydrogen.physik.uni-wuppertal.de/hyperphysics/hyperphysics/hbase/pespr.html)

The potential energy of a spring $U(x)$ is $\half k x^2$, where $k$ is the spring constant.

Now we could be interested on what is the probability to find the spring in a certain state!
This can be done by using the following Gibbs density distribution:

$$p(x) = \frac{1}{Z}\exp\left(-U(x)\right),$$

where $Z$ is the partition function of this system defined as

$$Z = \int_\Omega \exp\left(-U(x)\right)dx$$

where $\Omega$ is the domain of all the possible states of the system.
From a quick glance we can see that the highest probability is when $x=0$ and that this probability decreases when $x$ goes away from its rest position.
Physics don't like to have any efforts

The free energy is a complex concept which belongs mostly in thermodynamics. 
It has no real meaning in our example, since thermodynamics is the study of system with many objects.
Nonetheless we can still use its definition:

$$\mathcal{F} = -\log(Z)$$

Let's imagine that we have two springs (systems) with different constants $k_A$ and $k_B$, and associated potential energy function $U_A$ and $U_B$.
We can imagine that we control the variable $k$ and create a linear interpolation between the two creating a third intermediate system :

$$U_\lambda = U_A + \lambda(U_B - U_A)$$

Note that $U_0 = U_A$ and $U_1 = U_B$.

Now the most important result is that we can prove that the difference of free energy is given by:

$$\mathcal{F}_B - \mathcal{F}_A = \int_0^1 \expec{p_\lambda(x)}{U_B(x) - U_A(x)}d\lambda,$$

where $p_\lambda$ is defined by the potential energy $U_\lambda$.
~~~
<details>
    <summary> Proof :</summary>
~~~
    \begin{align}
    F_B - F_A =& F(1) - F(0) = \int_0^1 \frac{\partial F(\lambda)}{\partial \lambda}d\lambda\\
    \int_0^1 \frac{\partial F(\lambda)}{\partial \lambda}d\lambda &= -\int_0^1 \frac{\partial \log Z_\lambda}{\partial \lambda}d\lambda = -\int_0^1 \frac{1}{Z_\lambda}\frac{\partial Z_\lambda}{\partial \lambda}d\lambda \\
    &=\int_0^1 \int_\Omega \frac{\exp(-U_\lambda(x))}{Z_\lambda}\frac{\partial U_\lambda(x)}{\partial \lambda}dxd\lambda\\
    &=\int_0^1 \expec{p_\lambda(x)}{\frac{dU_\lambda(x)}{d\lambda}}d\lambda = \int_0^1 \expec{p(x)}{U_B(x) - U_A(x)}d\lambda
    \end{align}
~~~
</details>
~~~

What this result tell us is that we can compute the difference of free energy by computing the expectation of the difference of potentials when moving the system from one to another!
It is all very nice, but very abstract if you are not a physicist and what does it have to do with Bayesian inference?


## Using it in Bayesian Inference

Let's start with the basics: we have $x$, some hypothesis or parameters of a model, and $y$, some observed data! 
The **Bayes theorem** states that the posterior $p(x|y)$ is defined as:

$$p(x|y) = \frac{p(x,y)}{p(y)} = \frac{p(y|x)p(x)}{p(y)},$$

where $p(y|x)$ is the likelihood, $p(x)$ is the prior, we call $p(x,y)$ the joint distribution and finally $p(y)$ is the evidence.

We can rewrite it as an energy model, just as earlier!

$$p(x|y) = \frac{1}{Z}\exp\left(-U(x,y)\right)$$

where the potential energy $U(x,y) = -\log(p(x,y))$ is the negative log joint and the partition function $p(y)=Z=\int_\Omega \exp(-U(x,y))dx$ is the evidence.
We have now the direct connection that the free energy is nothing else than the log evidence.

We have a direct connection between energy models and Bayesian problems! 
Which leads us to the use of TI for Bayesian inference:

The most difficult part of the Bayes theorem comes from the fact that, except for simple cases, the posterior $p(x|y)$ as well as the evidence $p(y)$ are intractable.
$p(y)$ can be an important quantity to evaluate, if it's not for optimization of the model hyper-parameters, it can be simply for model comparisons.
However, computing $p(y)$ involves a high-dimensional integral with additional potential issues!
That's where Thermodynamic Integration can save the day!

## Creating a path from the prior to the posterior

Let's consider two systems again: prior : $U_A(x) = -\log p(x)$ and the joint distribution $U_B(x) = -\log p(x,y) = -\log p(y|x) - \log p(x)$. 
Our intermediate state is given by:

$$U_\lambda(x) = -\log p(x) + \lambda(-\log p(y|x) - \log p(x) - \log p(x)) = -\log p(x) - \lambda p(y|x)$$.

The normalized density derived from this potential energy is called a **power posterior** :

$$p_\lambda(x|y) = \frac{p(y|x)^\lambda p(y)}{Z_\lambda}$$,

which is a posterior for which we reduced the importance of the likelihood.

Performing the thermodynamic integration we derived earlier gives us :

$$ \mathcal{F}_B - \mathcal{F}_A = \log\int p(x,y)dx - \log \underbrace{\int p(x) dx}_{=1} = \log p(y) = \int_0^1 E_{p_\lambda(x)}[\log p(y|x)]d\lambda$$

Now let's put this to practice for a very simple use case of a Gaussian prior with a Gaussian likelihood

\begin{align}
p(x) = \mathcal{N}(x|\mu_p=10, \sigma_p=1)
p(y|x) = \mathcal{N}(y|x, \sigma_l=1)
\end{align}

We take $y = -10$ to make a clear difference between $U_A$ and $U_B$

Of course the posterior can be found analytically but this will help us to evaluate different approaches.
```julia:./code/thermint
using Plots, Distributions, LaTeXStrings
pyplot() # hide
default(lw = 2.0, legendfontsize = 15.0, labelfontsize = 15.0) # hide
σ_p = 1.0 # Define your standard deviation for the prior
σ_l = 1.0 # Define your standard deviation for the likelihood
μ_p = 10.0 # define the mean of your prior
prior = Normal(μ_p, σ_p) # prior distribution
likelihood(x) = Normal(x, σ_l) # function returning a distribution given x
y = -10.0
p_prior(x) = pdf(prior, x) # Our prior density function
p_likelihood(x, y) = pdf(likelihood(x), y) # Our likelihood density function
```
\output{./code/thermint}

Now we have all the parameters in place we can define the exact posterior and power posterior:
\begin{align}
p_\lambda(x|y) = \mathcal{N}(σ\lambda * \frac{\lambda y}{\sigma_l^2} + \frac{\mu_p}{\sigma_p^2}, \sigma_\lambda)
\sigma_\lambda = \left(\frac{1}{\sigma_p^2} + \frac{\lambda}{\sigma_l^2}\right)
\end{align}

```julia:./code/thermint
σ_posterior(λ) = sqrt(inv(1/σ_p^2 + λ / σ_l^2))
μ_posterior(y, λ) = σ_posterior(λ) * (λ * y / σ_l^2 + μ_p / σ_p^2)
posterior(y) = Normal(μ_posterior(y, 1.0), σ_posterior(1.0))
p_posterior(x, y) = pdf(posterior(y), x)
power_posterior(y, λ) = Normal(μ_posterior(y, λ), σ_posterior(λ))
p_pow_posterior(x, y, λ) = pdf(power_posterior(y, λ), x)
xgrid = range(-15, 15, length = 400)
plot(xgrid, p_prior , label = "Prior", xlabel = "x")
plot!(xgrid, x->p_likelihood(x, y),label =  "Likelihood")
plot!(xgrid, x->p_posterior(x, y), label = "Posterior")
savefig(joinpath(@OUTPUT, "distributions.svg")) # hide
plot(xgrid, [x->p_pow_posterior(x, y, λ) for λ in 0:0.2:1], # hide
    label = reshape(["λ = $λ" for λ in 0:0.2:1], 1, :), # hide
    title = "Power Posteriors", # hide
    xlabel = "x", # hide
    ylabel = L"p_\lambda(x|y)") # hide
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

Now we can start evaluating the integrand for multiple $\lambda$.
It is standard practice to use an irregular grid with step size $\epsilon_i = \left(\frac{i}{N}\right)^5$

```julia:./code/thermint
M = 100
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
using Trapz, Formatting
logpy = logpdf(posterior(y), y)
s_logpy = fmt(".2f", logpy)
TI_logpy = [trapz(λs[1:i], expec_λ[1:i]) for i in 1:length(λs)]
plot(λs, TI_logpy, label = "Therm. Int.", xlabel = L"\lambda") # hide
hline!([logpy], label = latexstring("\\log (p(y)) = ", s_logpy)) # hide
savefig(joinpath(@OUTPUT, "thermint.svg")) # hide
```
\output{./code/thermint}
\fig{./code/output/thermint.svg}

## Other approaches

Now that's great but how does it compare to other methods?
Well we want to compute the integral $p(y) = \int p(y|x)p(x)dx$. The most intuitive way is to sample from the prior $p(x)$ to perform a Monte-Carlo integration:

$$\int p(y|x)p(x)dx \approx \frac{1}{N}\sum_{i=1}^N p(y|x_i)$$

where $x_i \sim p(x)$.

```julia:./code/thermint
T = 10000
xs = rand(prior, T)
prior_logpy = [log(mean(pdf.(likelihood.(xs[1:i]), y))) for i in 1:T]
plot(1:T, prior_logpy, label = "Prior MC Integration", xlabel = "T") # hide
hline!([logpy], label = latexstring("\\log (p(y)) = ", s_logpy)) # hide
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
hline!([logpy], label = latexstring("\\log (p(y)) = ", s_logpy)) # hide
savefig(joinpath(@OUTPUT, "posterior_integration.svg")) # hide
```
\output{./code/thermint}
\fig{./code/output/posterior_integration.svg}

As you can see, it's not ideal either. If we now plot all our results together :

```julia:./code/thermint
bar(["log p(y)", "TI", "Prior", "Posterior"], [logpy, TI_logpy[end], prior_logpy[end], posterior_logpy[end]], lab="", ylabel="log Z") # hide
savefig(joinpath(@OUTPUT, "comparison_methods.svg")) # hide
```
\output{./code/thermint}
\fig{./code/output/comparison_methods.svg}

We can see that TI gives the most accurate result!
For a better estimate, it would be preferable to repeat these estimates multiple times, so I encourage you to try it yourself!
Of course other methods exist which I did not cover like Bayesian quadrature and others but this should give you a good idea on what is Thermodynamic Integration.