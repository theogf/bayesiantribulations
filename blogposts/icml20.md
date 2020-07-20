@def title = "ICML 2020"
@def hasmath = true
@def hascode = false
@def comment_issue_id = 1


# ICML 2020

So the [International Conference on Machine Learning (ICML)](https://icml.cc/) just ended and it was overall a great experience for me.

## My favorite talks

This blog post will aim at presenting some of the talks I really enjoyed watching. They are more or less in my order of preference. Clicking on the title will take you to the ICML website for which you need a registration. The link to the paper is accessible regardless.

- ### . [Black-Box Variational Inference as a Parametric Approximation to Langevin Dynamics](https://icml.cc/virtual/2020/poster/6629) Matt Hoffman, Yian Ma [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/5310-Paper.pdf)

  ![](/assets/icml20/bbvi_vs_langevin.png)

  Matt Hoffman strikes again! This work is more about a high-level understanding of the difference between *variational inference* (*VI*) and *Langevin dynamics*. Two topics I am highly interested in.

  Here is what I got from it: The general consensus is that sampling methods (like the [Metropolis Adjusted Langevin Algorithm](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm)) are less biased then VI methods like [Black-Box Variational Inference](https://arxiv.org/abs/1401.0118) (BBVI) since they converge to the true target distribution and are not limited to some parametrization. However VI methods have faster convergence and it is easier to assess that they have reached their optima.

  What Hoffman and Ma shows, is that the dynamics of both methods, which are essentially relying on sampling (one for estimating the distribution and one for estimating its gradients) or extremely similar. This kind of results appear more and more in recent papers like the paper [Stochastic Gradient and Langevin Processes](https://proceedings.icml.cc/static/paper_files/icml/2020/1074-Paper.pdf) from the same conference.

  Based on this equivalence they show that MALA can be interpreted as some kind of non-parametric BBVI. And that if one does not wait for the convergence of the sampler, decent results can be obtained already. It turns out that for some cases that these results are better or equal than the ones of BBVI. This is definitely thought-provoking! Finally, they argue that with parallelization it is now easier to run multiple chains and that this can help to outperform VI approaches.

  I really recommend watching the talk as on top of the interesting work, Matt Hoffman is an amazing speaker!

- ### 2. [All in the (Exponential) Family: Information Geometry and Thermodynamic Variational Inference](https://icml.cc/virtual/2020/poster/6234) Rob Brekelmans, Vaden Masrani, Frank Wood, Greg Ver Steeg, Aram Galstyan [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/2826-Paper.pdf)

  ![](/assets/icml20/tvo.png)

  This paper is based on the [Thermodynamic Variational Objective paper](https://arxiv.org/abs/1907.00031) which have yet to read! But the general idea it that we replace the ELBO by a series of "tempered" ELBO, where the variational distribution is a mixture between a variational family $q_\phi$ and the joint distribution $p(z,x)$.

  $$\pi_\beta(z|x) \propto  q_\phi(z|x)^{1-\beta}p_\theta(x, z)^\beta$$

  By progressively going from $\beta=0$ to $\beta = 1$ we go from the variational distribution to the true posterior. This gives us a much more exact estimate of the log-evidence. For those interested in this approach there is [this blog post](http://eriqande.github.io/sisg_mcmc_course/thermodynamic-integration.nb.html) but I will probably try to write my own.

  Now one normally integrate over a couple of $\beta$ but they show that by choosing the right $\beta$ one can already improve the ELBO value. To do this they rewrite $\pi_\beta$ as an exponential family :

  $$\pi_\beta(z|x) = q_\phi\exp\left[\beta \log\frac{p_\theta(x,z)}{q_\phi(z|x)} - \log Z_\beta(x)\right]$$

  To be honest I am not sure how this works for $\beta$ :slightly_smiling_face:

  They also show that one can when integrating from $\beta=0$ to $\beta=1$, one can automatically find the best step-size.

  From the discussions one of the pit-falls of the methods seems to be the dimensionality of the problem. The expectations computed require importance sampling, which is known to be weak in high-dimensions.

- ### 3. [Involutive MCMC: One Way to Derive Them All](https://icml.cc/virtual/2020/poster/6476) Kirill Neklyudov, Max Welling, Evgenii Egorov, Dmitry Vetrov, [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/4339-Paper.pdf) 

  ![](/assets/icml20/iMCMC.png)

  Generalization papers are always interesting because they bring a higher level of what some algorithms do. In this talk they aim at bringing a lot of **Markov Chain Monte Carlo** (MCMC) algorithms [under one hood](https://vignette.wikia.nocookie.net/lotrfanon/images/b/b8/7bcdf0cf3d97a4467f082850758c527d.jpg/revision/latest/scale-to-width-down/220?cb=20181224143216).

  The basic idea of MCMC is to sample from a distribution by moving in the variable domain. Every move is subject to a rejection step, based on probability of the sample and the probability of the move. The samples collected will then be assumed to come from the target distribution. More formally the chain is

  $$x_{i+1} \sim t(x|x_i)$$

  where the condition is that the **kernel** is density invariant : $\int t(x'|x)p(x)dx = p(x')$

  One such kernel is $t(x'|x) = \delta(x'-f(x))$ where $f(x)$ has to respect the condition

  $$ p(x) = p(f(x))\left|\frac{\partial f}{\partial x}\right| = p(f^{-1}(x))\left|\frac{\partial f^{-1}}{\partial x}\right|$$

  When adding the acceptance step one gets the following kernel

  $$t(x'|x) = (x' = f(x))\min\left[1, \frac{p(f(x))}{p(x)}\left|\frac{\partial f}{\partial x}\right|\right] +  (x' = x)\left( 1 - \min\left[1, \frac{p(f(x))}{p(x)}\left|\frac{\partial f}{\partial x}\right|\right]\right)$$

  Now the problem with getting a $f$ satisfying this condition is that you will end up cycling between two locations. To solve this problem we need an additional auxiliary variable $v$. 

  The **involution** restriction is now relaxed to $f(x, v) = f^{-1}(x, v)$ . If we take the Metropolis Hasting algorithm this would be sample $v~p(v|x)$ our proposal. For example for the random walk algorithm, this would mean sampling from a Gaussian centered in $x$ . Then $f(x, v) = [v,x]$ (notice the permutation). The acceptance rate gives then $P = \min \left\{ 1, \frac{p(v,x)}{p(x,v)}\right\}$

  Following this definition, they list a series of tricks including additional auxiliary augmentations, additional involutions and deterministic map.
  
  This talk really fascinated me as it really gives openings to create new and more efficient sampling algorithms!

## Other talks

Here are other presentations that really caught my attention and that I will probably explore later

- [Scalable Exact Inference in Multi-Output Gaussian Processes](https://icml.cc/virtual/2020/poster/6430) Wessel Bruinsma, Eric Perim Martins, William Tebbutt, Scott Hosking, Arno Solin, Richard E Turner [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/4027-Paper.pdf)

  General framework for multi-output GPs with a smart projection to reduce the complexity of the problem. It looks very sound  theoretically 

- [Sparse Gaussian Processes with Spherical Harmonic Features](https://icml.cc/virtual/2020/poster/5898) https://icml.cc/virtual/2020/poster/5898[[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/906-Paper.pdf) 

  Augmenting the data with an additional parameter (same for all data) and then projecting the data on an hypersphere to use spherical harmonics as inducing points

- [Efficiently sampling functions from Gaussian process posteriors](https://icml.cc/virtual/2020/poster/6461) James Wilson, Slava Borovitskiy, Alexander Terenin, Peter Mostowsky, Marc Deisenroth [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/4232-Paper.pdf) 

  By separating the contribution of the prior and the data, one can sample more efficiently (linear time!) by using random fourier features for the prior.

- [Automatic Reparameterisation of Probabilistic Programs](https://icml.cc/virtual/2020/poster/5804) Maria Gorinova, Dave Moore, Matt Hoffman [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/271-Paper.pdf) 

  By creating an interpolation between the centered version and non-centered version of a graphical model one can find the optimal representation for sampling/inference.

- [Stochastic Differential Equations with Variational Wishart  Diffusions](https://icml.cc/virtual/2020/poster/6149) Martin Jørgensen, Marc Deisenroth, Hugh Salimbeni [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/1074-Paper.pdf) 

  When one wants to represent noise with a non-stationary full covariance depending on time a Wishart process helps to determine it. The technique use GPs for the cholesky parameters and present a low-rank approximation for efficiency

- [Stochastic Gradient and Langevin Processes](https://icml.cc/virtual/2020/poster/5923) Xiang Cheng, Dong Yin, Peter Bartlett, Michael Jordan [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/1074-Paper.pdf) 

  Show equivalence between Stochastic Gradient Descent (SGD) and Langeving processes (them again!)

- [Variance Reduction and Quasi-Newton for Particle-Based Variational Inference](https://icml.cc/virtual/2020/poster/6650) Michael Zhu, Chang Liu, Jun Zhu [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/5434-Paper.pdf) 

  Preconditioners for optimization are not straight forward in the particle-based optimization framework. This work aims at solving that

- [Learning the Stein Discrepancy for Training and Evaluating Energy-Based Models without Sampling](https://icml.cc/virtual/2020/poster/6649) Will Grathwohl, Kuan-Chieh Wang, Jörn Jacobsen, David Duvenaud, Richard Zemel  [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/5430-Paper.pdf)

  Stein Discrepancy is complicated to use, and the kernelized versions has a lot of flaws, instead they propose to learn a constrained neural net to replace the kernel and obtain better results

- [Non-convex Learning via Replica Exchange Stochastic Gradient MCMC](https://icml.cc/virtual/2020/poster/6023) Wei Deng, Qi Feng, Liyao Gao, Faming Liang, Guang Lin [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/1632-Paper.pdf)

  One runs two processes with very different stepsize, there is then a chance for the sampler to switch between the two chains, allowing for a compromise between exploration and exploitation.

- [Handling the Positive-Definite Constraint in the Bayesian Learning Rule](https://icml.cc/virtual/2020/poster/6821) Wu Lin, Mark Schmidt, Emti Khan [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/6575-Paper.pdf) 

  In order to optimize a positive-definite matrix, they propose to add a corrective term (through information geometry) to ensure the PD constraint is kept.

## The online format

Due to COVID the conference was naturally online, which was for me a double-edged sword. On one hand it is amazing to be able to parse the presentations one by one and to take time to understand each topic. On the other hand, the lack of physical presence made it quasi-impossible to network with other people.

I went to a few poster sessions, aka local Zoom meetings, and it was definitely a slightly awkward experience. You suddenly end up in front of the multiple authors. It definitely puts a lot more pressure. If there is one point that needs to be improved it is this one!
