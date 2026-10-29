# Derivations: occurrence error proxies of the iETS models

Study `iets-occurrence-bias`. Each section supports one formal hypothesis of the lab record;
`derivations_check.py` verifies every identity symbolically (sympy) and every inequality on a
high-precision grid. Notation follows Svetunkov and Boylan (2023), henceforth SB23.

## 1. Setting

Demand occurrence $o_t \sim \text{Bernoulli}(p_t)$. The occurrence part of iETS keeps a
positive state $\mu$ and maps it to a probability:

| Model | Probability | State updated |
|---|---|---|
| iETS_O (odds ratio) | $p = \mu_a/(1+\mu_a)$ | $\mu_a$ |
| iETS_I (inverse odds ratio) | $p = 1/(1+\mu_b)$ | $\mu_b$ |
| iETS_G (general) | $p = \mu_a/(\mu_a+\mu_b)$ | $\mu_a$ and $\mu_b$ |

With a multiplicative occurrence error the state is updated as
$\mu_t = \mu_{t-1}(1+\alpha e_t)$; with an additive error the state is kept on the log scale,
$\log\mu_t = \log\mu_{t-1} + \alpha e_t$. The error $e_t$ is not observable, so a proxy built
from $o_t$ and the one-step-ahead probability $\hat p_t$ is used in its place.

**Old proxies** (SB23, eqs. 24-25 and Appendix A; smooth's `occurrenceError()`). With the
residual $v_t = o_t - \hat p_t \in (-1,1)$ and $u_t = (1+v_t)/2 \in (0,1)$:

$$1+e_{a,t} = \frac{u_t}{1-u_t} = \frac{1+v_t}{1-v_t}, \qquad
  1+e_{b,t} = \frac{1-u_t}{u_t} = \frac{1-v_t}{1+v_t}.$$

The additive-error versions use $e = \log(\cdot)$ of the same ratios.

**Filter of interest.** Throughout, $o_t \sim \text{Bernoulli}(p)$ with $p \in (0,1)$ fixed,
and expectations are over $o_t$ given the current $\hat p_t = x$. The **expected update**
(drift) of the state is $\mathrm{E}[\mu_t - \mu_{t-1} \mid \mu_{t-1}] = \alpha\mu_{t-1}\,h(x)$
with $h(x) = \mathrm{E}[e_t \mid \hat p_t = x]$ (multiplicative), and
$\mathrm{E}[\log\mu_t - \log\mu_{t-1} \mid \mu_{t-1}] = \alpha\,h(x)$ (additive). A
**zero-drift point** is an $x^*$ with $h(x^*) = 0$.

## 2. The old proxies have a non-zero expected update at the true probability (H5)

The residual is unbiased: at $x = p$, $\mathrm{E}[v_t] = 0$. The proxies are convex
transformations of it, so their means are not those of an unbiased error. At $x = p$,
$u_t = (2-p)/2$ with probability $p$ and $u_t = (1-p)/2$ with probability $1-p$:

$$\mathrm{E}[1+e_a] = p\,\frac{2-p}{p} + (1-p)\frac{1-p}{1+p}
  = 1 + (1-p) + \frac{(1-p)^2}{1+p} > 1,$$

$$\mathrm{E}[1+e_b] = p\,\frac{p}{2-p} + (1-p)\frac{1+p}{1-p}
  = 1 + p + \frac{p^2}{2-p} > 1.$$

Both exceed 1 for every $p \in (0,1)$: at the true probability, $\mu_a$ (iETS_O) is pushed
up, so $\hat p$ rises above $p$, and $\mu_b$ (iETS_I) is pushed up, so $\hat p$ falls below
$p$. The expansion $\frac{1+v}{1-v} = 1 + 2v + 2v^2 + 2v^3 + \dots$ shows the cause: the
linear term has mean zero, and the even terms ($\mathrm{E}[v^2] = p(1-p)$ at the truth) do not.

With an additive error,
$$g(p) := \mathrm{E}[\log(1+e_a)] = p\log\frac{2-p}{p} + (1-p)\log\frac{1-p}{1+p},
  \qquad \mathrm{E}[\log(1+e_b)] = -g(p).$$
$g(1-p) = -g(p)$, so $g(1/2) = 0$; $g > 0$ on $(0, 1/2)$ (verified on the grid). The update is
zero at the truth only for $p = 1/2$, and otherwise moves $\hat p$ towards $1/2$.

## 3. Zero-drift points of the old filters (H6)

**iETS_I, multiplicative.** With $x = \hat p$,
$$h_b(x) = p\frac{x}{2-x} + (1-p)\frac{1+x}{1-x} - 1 = \frac{-2\,(x^2 - 2x + p)}{(2-x)(1-x)}.$$
The only root in $(0,1)$ is $x^* = 1 - \sqrt{1-p}$, and $x^* < p$ because
$\sqrt{1-p} > 1-p$. For $x < x^*$, $h_b < 0$: $\mu_b$ falls and $\hat p$ rises; for
$x > x^*$ the reverse. $x^*$ is the unique stable zero-drift point.

**iETS_O, multiplicative.**
$$h_a(x) = p\frac{2-x}{x} + (1-p)\frac{1-x}{1+x} - 1 = \frac{2\,(p - x^2)}{x(1+x)},$$
with the unique root $x^* = \sqrt{p} > p$, stable by the same sign argument.

**Both models, additive error.** $h_b(x) = p\log\frac{x}{2-x} + (1-p)\log\frac{1+x}{1-x}$
and $h_a(x) = -h_b(x)$ evaluated at the probability of the respective model, so both have the
same zero-drift point: the root $x^*$ of
$$p\log\frac{x}{2-x} + (1-p)\log\frac{1+x}{1-x} = 0,$$
which is unique in $(0,1)$, stable, equals $1/2$ at $p = 1/2$ and lies strictly between $p$
and $1/2$ otherwise (verified on the grid).

**From zero-drift points to forecasts.** For a constant gain $\alpha$, the iterates of a
stochastic approximation concentrate, as $\alpha \to 0$, around the stable zero of its mean
dynamics $\dot\mu = \mu\,h(p(\mu))$ (Benveniste, Métivier and Priouret 1990; Kushner and Yin
2003, ch. 8). The stationary mean of $\hat p_t$ should therefore converge to $x^*$ as
$\alpha \to 0$, not to $p$: the bias of the old filters is $x^* - p$, of order one, and does
not vanish with a small smoothing parameter. This step is an assumption (the theorems' regularity
conditions are not verified here); the simulations test its consequence (H8).

Since the demand-size part of iETS is mean-unbiased, a biased $\hat p$ biases the point forecast
$\hat y = \hat p\hat z$ by the same factor: $x^*/p$, e.g. $(1-\sqrt{1-p})/p \to 1/2$ as
$p \to 0$ for iETS_I.

**Contrast with Croston's method.** Croston's $1/\hat q$ is biased upwards by Jensen's
inequality applied to an SES of the intervals ("inversion bias"; Syntetos and Boylan 2001, 2005).
That bias is of order $\alpha$ and vanishes as $\alpha \to 0$.

## 4. The score proxy (H7)

The occurrence log-likelihood of one observation is $\ell = o\log p + (1-o)\log(1-p)$ (the
occurrence part of SB23, eq. 29), with $\partial\ell/\partial p = (o-p)/\{p(1-p)\}$.

| Model | $\partial p/\partial\log\mu$ | Score $\partial\ell/\partial\log\mu$ |
|---|---|---|
| iETS_O, $\mu_a$ | $p(1-p)$ | $o - p$ |
| iETS_I, $\mu_b$ | $-p(1-p)$ | $p - o$ |
| iETS_G, $\mu_a$ and $\mu_b$ | $p(1-p)$, $-p(1-p)$ | $o - p$, $p - o$ |

The **score proxy** replaces the unobservable error with this score at $\hat p_t$:
$$e_{a,t} = o_t - \hat p_t, \qquad e_{b,t} = \hat p_t - o_t,$$
used as $\log\mu_t = \log\mu_{t-1} + \alpha e_t$ (additive error) and as
$\mu_t = \mu_{t-1}(1+\alpha e_t)$ (multiplicative error, the first-order equivalent:
$\log(1+\alpha e) = \alpha e + O(\alpha^2)$). The additive version is a score-driven
(generalised autoregressive score, or dynamic conditional score) filter with unit scaling
(Creal, Koopman and Lucas 2013; Harvey 2013).

Properties:
1. $\mathrm{E}[e \mid \hat p_t = x] = \pm(p - x)$ (sign as in the table), so the expected
   update is zero exactly at $x = p$, for every $p$, and its sign always moves $\hat p$
   towards $p$: $p$ is the unique, stable zero-drift point.
2. $e \in (-1,1)$, so $1+\alpha e > 0$ for every $\alpha \in (0,1]$: the multiplicative state
   stays positive with no truncation constant.
3. The likelihood (SB23, eq. 29) is unchanged; only the filter is.
4. Unit scaling is chosen because it keeps the proxy bounded; scaling by the inverse Fisher
   information, $\{p(1-p)\}^{-1}$, would make it unbounded as $p \to 0$ or $1$.

By the argument of §3, the stationary mean of $\hat p_t$ should converge to $p$ as
$\alpha \to 0$ (H10). What remains at a positive $\alpha$ should be of order $\alpha$: the
log-state of the multiplicative version drifts by $\mathrm{E}[\log(1+\alpha e)] \approx
\alpha\mathrm{E}[e] - \alpha^2\mathrm{E}[e^2]/2$, and $\hat p$ is a nonlinear function of a
fluctuating state. That residual bias is not derived here; H11 compares it with the old
proxies' at every $\alpha$.

iETS_D ($p = \mu_a$, SB23 eq. 21) has score $(o-p)/(1-p)$ with respect to $\log\mu_a$; it is
outside this study.

## References

- Benveniste, A., Métivier, M. and Priouret, P. (1990). *Adaptive Algorithms and Stochastic
  Approximations*. Springer. doi:10.1007/978-3-642-75894-2
- Creal, D., Koopman, S.J. and Lucas, A. (2013). Generalized autoregressive score models with
  applications. *Journal of Applied Econometrics* 28(5), 777-795. doi:10.1002/jae.1279
- Croston, J.D. (1972). Forecasting and stock control for intermittent demands. *Operational
  Research Quarterly* 23(3), 289-303.
- Harvey, A.C. (2013). *Dynamic Models for Volatility and Heavy Tails*. Cambridge University
  Press. doi:10.1017/CBO9781139540933
- Kushner, H.J. and Yin, G.G. (2003). *Stochastic Approximation and Recursive Algorithms and
  Applications*, 2nd ed. Springer. doi:10.1007/b97441
- Svetunkov, I. and Boylan, J.E. (2023). iETS: State space model for intermittent demand
  forecasting. *International Journal of Production Economics* 265, 109013.
  doi:10.1016/j.ijpe.2023.109013
- Syntetos, A.A. and Boylan, J.E. (2001). On the bias of intermittent demand estimates.
  *International Journal of Production Economics* 71(1-3), 457-466.
  doi:10.1016/S0925-5273(00)00143-2
- Syntetos, A.A. and Boylan, J.E. (2005). The accuracy of intermittent demand estimates.
  *International Journal of Forecasting* 21(2), 303-314. doi:10.1016/j.ijforecast.2004.10.001
