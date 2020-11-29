@def title = "Automatic Conditional Conjugacy for Gaussian Processes"
@def hasmath = true
@def comment_section = true
@def hacode = true

# Automatic Conditional Conjugacy for Gaussian Processes

Last summer our paper ["Automated Augmented Conjugate Inference for Non-conjugate Gaussian Process Models"](https://arxiv.org/abs/2002.11451) written with Florian Wenzel and Manfred Opper (my supervisor) got accepted at AISTATS 2020.
This is a paper I am particularly proud of, as it contains both a beautiful theory and a direct application.
Although you can find the video of the presentation here, I thought I would write a small blog post to give a light approach to it.

~~~
<div id="presentation-embed-38930226"></div>
<script src='https://slideslive.com/embed_presentation.js'></script>
<script>
    embed = new SlidesLiveEmbed('presentation-embed-38930226', {
        presentationId: '38930226',
        autoPlay: false, // change to true to autoplay the embedded presentation
        verticalEnabled: true
    });
</script>
~~~

## The problem

Over the past years, my research has been focused on how to modify likelihood functions to make them nicer to work with when you have a Gaussian prior.
Mostly, I worked with previous work which found a correspondence between for example the logistic likelihood and the Polya-Gamma variable.

Let's give a simple example. You should be familiar with the logistic function:

$$\sigma(f) = \frac{1}{1+\exp(-x)}$$

```julia:.
using Plots # hide
σ(f) = inv(1 + exp(-f))
plot(-5:0.01:5, σ, lab="", xlabel = "f", ylabel = "σ(x)") # hide
savefig(joinpath(@OUTPUT, "logistic.svg")) # hide
```
\output{.}
\fig{./logistic.svg}

It is heavily used for binary classification, since $\sigma(f)\in [0, 1]$, for example for Gaussian Process classification, one can use $p(y_i|f_i) = \text{Bernoulli}(y|\sigma(f_i))$.

The problem with such a link is that working with Gaussian priors (as in Gaussian Processes) makes the posterior untractable.

The different solution are Variational Inference (VI), where one look for the closest distribution $q(f)$ to the true posterior $p(f|y)$ by minimizing the KL divergence, sampling and others.

The problem with VI is that one needs to compute the expectation of the log-likelihood $\expec{q(f)}{\log p(y|f)}$ and get its gradient given the variational parameters.
Often this integral is intractable as well.
One can use quadrature, sampling and other approaches, but they can turn out expensive and/or inaccurate.

What we propose is to rewrite the likelihood into a form where the integral becomes analytically tractable.

## Scale Mixture of Gaussians

Let's introduce a family of functions called scale mixtures of Gaussians.
These can be written as :

$$f(x) \propto \int_0^\infty\mathcal{N}(0, \omega)p(\omega)d\omega.$$

You can imagine it as being an infinite weighted sum of Gaussians with different variance $\omega$.
A well known example is for instance the Student-T likelihood :

```julia:.
using Distributions
nu = 2.0
studentt = TDist(nu)
pomega = InverseGamma(nu, nu)
plot(-5:0.01:5, x->pdf(studentt, x), label="Student-T")

plot!()
```