<!-----------------------------------------------------
Add here global page variables to use throughout your
website.
The website_* must be defined for the RSS to work
------------------------------------------------------->
@def website_title = "Bayesian Tribulations"
@def website_descr = "Tribulations in a Bayesian life"
@def website_url   = "https://theogf.github.io/bayesiantribulations/"
@def prepath = "bayesiantribulations"

@def author = "Theo Galy-Fajou"

<!-----------------------------------------------------
Add here global latex commands to use throughout your
pages. It can be math commands but does not need to be.
For instance:
------------------------------------------------------->
\newcommand{\reals}{\mathbb R}
\newcommand{\scal}[1]{\langle #1 \rangle}
\newcommand{\deriv}[2]{\frac{d#1}{d#2}}
\newcommand{\bx}{\boldsymbol{x}}
\newcommand{\derivk}[1]{\frac{dk(\boldsymbol{x},\boldsymbol{x}')}{d#1}}
\newcommand{\norm}[1]{||#1||}
\newcommand{\KL}{\text{KL}}
\newcommand{\tr}{\text{tr}}
\newcommand{\diag}{\text{diag}}
\newcommand{\expec}[2]{\mathbb{E}_{#1}\left[#2\right]}


<!-- Put a box around something and pass some css styling to the box
(useful for images for instance) e.g. :
\style{width:80%;}{![](path/to/img.png)} -->
\newcommand{\style}[2]{~~~<div style="!#1;margin-left:auto;margin-right:auto;">~~~!#2~~~</div>~~~}
