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
## Scale mixture of Gaussian

Over the past years, my research has been focused on how to modify likelihood functions to make them nicer to work with when you have a Gaussian prior.
Mostly, I worked with previous work which found a correspondence between for example the logistic likelihood and the Polya-Gamma variable 