+++
title = "Thermodynamic Integration"
hasmath = true
comment_section = true
blog_post = true
hascode = true
+++

# The Kalman Filter and the Unscented Transformation

Let's start with a topic so widely used in signal processing: the Kalman filter.
To understand how this works we are going to follow a simple example along the whole post:
**mouse movement tracking**.
Let's assume that every small time step $t$ we measure where the mouse is and save this information as $Y_t= [y^1_t, y^2_t]$.
We want to smooth out the movement of the mouse and infer a general trajectory of the mouse that we define as $X_t=


The basic principle of the Kalman filter is to consider the model as a **hidden Markov Chain model**.



Unscented Kalman filter:

- Transform the sigma points
- Estimate the new mean and covariance from the transformed points
- Create again new sigma points from new parameters to predict observations-
- Compute differences
- Update the mean/covariance