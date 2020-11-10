<!--
Add here global page variables to use throughout your
website.
The website_* must be defined for the RSS to work
-->
@def website_title = "Bayesian Tribulations"
@def website_descr = "Tribulations in a Bayesian life"
@def website_url   = "https://theogf.github.io/bayesiantribulations/"
@def prepath = "bayesiantribulations"
@def github_account = "theogf/bayesiantribulations"
@def require_cookie_consent = true
@def author = "Theo Galy-Fajou"
@def comment_section = false

<!-- @def mintoclevel = 2 -->
<!--
Add here files or directories that should be ignored by Franklin, otherwise
these files might be copied and, if markdown, processed by Franklin which
you might not want. Indicate directories by ending the name with a `/`.
-->
@def ignore = ["node_modules/"]

<!-----------------------------------------------------
Add here global latex commands to use throughout your
pages. It can be math commands but does not need to be.
For instance:
------------------------------------------------------->
\newcommand{\deriv}[2]{\frac{d#1}{d#2}}
\newcommand{\derivk}[1]{\frac{dk(\boldsymbol{x},\boldsymbol{x}')}{d#1}}
\newcommand{\norm}[1]{||#1||}
\newcommand{\KL}{\text{KL}}
\newcommand{\tr}{\text{tr}}
\newcommand{\diag}{\text{diag}}
<!-- \newcommand{\expec}[2]{\mathbb{E}_{#1}\left[#2\right]} -->
