# Independent proof check: intermittent-likelihood claimset

Checked by an AI agent (Claude Sonnet 5.5 in Claude Code), not an author, on 2026-11-11. Method: for each listed claim I read its statement, the definitions it marks, the claims it depends on and assumes (taken as given), the evidence steps and the matching passage of the manuscript `int-draft.tex`; I re-derived the result by hand from the premises and then spot-checked the identities numerically with Python (numpy only; scipy is not installed, so integrals were done on dense grids and Gauss-type sums, and Monte Carlo used numpy's `default_rng`). The scripts are `chk1.py`, `chk2.py` and `chk3.py` in this folder; every number below comes from running them. I did not consult Svetunkov and Boylan (2023) itself, so what that paper prints is not verified here.

Summary: all ten claims are confirmed; five are confirmed only in part (C2, C5, C10, C14 and C19), none is disputed. Several evidence objects have gaps listed per claim.

| Claim | Evidence | Outcome | Extent |
|---|---|---|---|
| C1 | E1 | confirmed | full |
| C2 | E2 | confirmed | partial |
| C5 | E6 | confirmed | partial |
| C6 | E9 | confirmed | full |
| C8 | E10 | confirmed | full |
| C10 | E12 | confirmed | partial |
| C12 | E8 | confirmed | full |
| C14 | E15 | confirmed | partial |
| C16 | E7 | confirmed | full |
| C19 | E13 | confirmed | partial |

---

## C1 (E1): observed-data likelihood of intermittent demand

**Outcome: confirmed, full.**

Derivation. With $o_t\perp z_t$, $z_t$ continuous and positive: $P(y_t=0\mid Y_{t-1})=(1-p_t)+p_tP(z_t=0)=1-p_t$, and for $y>0$ the density of $y_t$ is $p_t f(y\mid Y_{t-1})$. The joint density of the observed data is $\prod_t f(y_t\mid Y_{t-1})$ (prediction-error decomposition), and each factor is one of those two. Taking logs gives the stated sum. The unobserved $z_t$ never enter, and the size part $\sum_{O_t=1}\log f(z_t\mid Y_{t-1};\theta)$ and the occurrence part $\sum\log p_t+\sum\log(1-p_t)$ depend on disjoint parameters, so they separate. Conditioning on $Y_{t-1}$ at a zero adds nothing about the size, because $o_t=0$ carries no information on $z_t$ (independence).

Numerical check (`chk1.py`). ETS(A,N,N), $\alpha=0.4$, $T=6$, periods 3 and 5 unobserved. I integrated the complete-data Gaussian density over the two unobserved sizes on a 1201x1201 grid and compared with the Kalman-filter predictive-density sum (the $f(z_t\mid Y_{t-1})$ of the claim): brute-force $-5.713299723643796$ against filter $-5.713299723643553$ (difference $2.4\times10^{-13}$). So the size part equals the integral of D3.

Gaps and typos.
- E1.2 and E1.1 need $z_t>0$ (so that $y_t>0\iff o_t=1$); C32 states only $P(z_t=0)=0$. Positivity of sizes should be an assumption.
- C31 states independence of $o_t$ and $z_t$ at the same period. The factorisation needs the whole occurrence process (including $p_t$, which may depend on $Y_{t-1}$) to carry no information about the sizes, or $p_t$ to depend on the occurrence history only. State it.
- E1.3 does not show the sum over $t$ (log of the product) and does not mention that $\theta$ and $\lambda$ (size and occurrence parameters) are disjoint, which is what "separate" means.
- The equality with D3 (integral of the complete-data likelihood) is only noted in the manuscript proof ("Equivalently ..."), not in E1.

---

## C2 (E2): stationary points of the entropy-augmented likelihood

**Outcome: confirmed, partial.** Verified: the gradient facts and, for a normal size with fixed location, the actual difference of the stationary points. Not verified: that the E-step in the dynamic (state-dependent) iETS has exactly the cross-entropy form used in E2.1.

Derivation. For densities $f^{(k)}=f_z(\cdot\mid\theta^{(k)})$ and $f_\theta=f_z(\cdot\mid\theta)$:
$$\int f^{(k)}\log f_\theta\,dz=-H_z(\theta^{(k)})-\mathrm{KL}(f^{(k)}\|f_\theta),$$
and $\partial_\theta\int f_\theta\log f_{\theta'}\,dz\big|_{\theta'=\theta}$ is $\int f_\theta\partial_\theta\log f_\theta\,dz=\partial_\theta\int f_\theta=0$ (score has mean zero). So $\nabla Q(\theta\mid\theta^{(k)})|_{\theta^{(k)}}=\nabla\ell(\theta^{(k)})$ (Fisher's identity; the cited EM reference), and EM fixed points are stationary points of $\ell$ of D3. The entropy term is $-H_z(\theta)=\int f_\theta\log f_\theta$, which sets $\theta^{(k)}=\theta$ before differentiating, so it contributes $-\partial_\theta H_z\neq0$ to the score: $\nabla\ell_H=\nabla\ell_{\text{size}}-\sum_{O_t=0}\partial_\theta H_z$.

Numerical checks (`chk3.py`, normal sizes with fixed location, $T_1=60$, $T_0=40$, $\sum e^2=S$):
- the cross-entropy $\int N(0,1.3)\log N(0,s_t)$ equals $-H-\mathrm{KL}$: $-1.5994310150$ against $-1.5994310151$ (grid integral against closed form);
- derivative of the cross-entropy in the variance at $s=s_k$: $0.0$; derivative of $-H$ at the same point: $-0.3846153845$ (equals $-1/(2\cdot1.3)$);
- argmax of $\ell_H$ in $\sigma^2$: $1.9910$ against $S/T=1.99084$; argmax of the observed-data likelihood and EM fixed point: $3.3181$ against $S/T_1=3.31806$. The stationary points differ, with the entropy version shrunk by $T_1/T=0.6$ (Corollary in the manuscript).

Gaps and typos.
- E2 shows that the extra gradient is non-zero, but does not itself conclude that the stationary points differ ($\nabla\ell_H=0$ could in principle still coincide with $\nabla\ell=0$ at special parameter values). The manuscript's Corollary (normal scale) is the actual demonstration; add it, or its general form $\nabla\ell(\theta^*)=\sum\partial_\theta H_z(\theta^*)$, to E2.
- In the state-dependent model the posterior of an unobserved size given the observed data is not $f_z(\cdot\mid\theta^{(k)})$ with the one-step location, since later observations inform it through the states. E2.1's cross-entropy form is exact only for sizes conditionally independent given the parameters (the static-location case). Also $\ell_H$ uses the zero-error one-step densities for the observed sizes, not $f(z_t\mid Y_{t-1})$, which is a second source of difference (C5), absent from the stated reason.
- E2.1 cites D4 (the objective) as "definition of the E-step"; it should cite the EM definition. E2.3 uses the symbol $\ell$ without definition (it is the log of D3); `B1` is a bibliography key, not a claim.

---

## C5 (E6): the zero-error recursion maximiser is not the MLE

**Outcome: confirmed, partial.** Verified: the objectives differ as stated (variance $\sigma^2f_t$ and gain $\mathbf k_t$) and, numerically for ETS(A,N,N), that the maximisers differ and the zero-error one is inconsistent. Not proved in E6: that differing objectives give differing maximisers.

Derivation. From D8 started at $\mathbf P=0$, after $j-1$ unobserved periods $\mathbf P=\sum_{i=1}^{j-1}\mathbf F^{i-1}\mathbf g\mathbf g'\mathbf F'^{\,i-1}$, so $f=1+\mathbf w'\mathbf P\mathbf w=1+\sum_{i<j}c_i^2$ with $c_i=\mathbf w'\mathbf F^{i-1}\mathbf g$ (E6.1 correct). If $c_1\neq0$ then $f>1$ at the next observed period. The exact conditional law is $\mathcal N(\mathbf w'\mathbf a_t,\sigma^2f_t)$ (C4) with update by $\mathbf k_t=(\mathbf F\mathbf P\mathbf w+\mathbf g)/f$; D5 uses $\sigma^2$ and $\mathbf g$. Hence the two log-likelihoods are different functions of $\theta$.

Numerical check (`chk1.py`). ETS(A,N,N), true $\alpha=0.3$, $\sigma=1$, $T=20000$, each period observed with probability 0.5 (and 0.3), initial level fixed at its true value; grid search over $\alpha$ (step 0.005) for the exact filter likelihood (concentrated variance) and for the zero-error concentrated likelihood, 6 seeds each. Exact MLE: mean 0.3008 (0.5 observed), 0.3017 (0.3 observed). Zero-error maximiser: mean 0.3967 (all six between 0.385 and 0.405) and 0.4750 (0.470 to 0.480). So the zero-error maximiser is clearly different and biased upwards, as the claim's gain argument suggests.

Gaps and typos.
- E6.3 concludes "its objective differs"; the claim is about the maximiser. Needed: an argument that the expected score of the zero-error objective is not zero at the true parameter (or a counterexample class), as the numerical check shows for ETS(A,N,N). Add to E6.
- Hypotheses missing from the claim: it needs $c_i\neq0$ for some $i<j$ (observability of $\mathbf g$ through $\mathbf w'\mathbf F^{i-1}$). Counterexample to "g different from 0 suffices": $\mathbf F=\mathbf I_2$, $\mathbf w=(1,0)'$, $\mathbf g=(0,\beta)'$: $c_i=0$ for all $i$, $f_t=1$, $\mathbf k_t=\mathbf g$, and the two recursions coincide. The manuscript says "$c_1\neq0$" in its proof but "$\mathbf g\neq0$" in the statement.
- E6.2 "$\mathbf k_t\neq\mathbf g$ while $\mathbf P_t\neq0$" is false in general: for ETS(A,N,N), $k=(P+\alpha)/(P+1)=\alpha$ for any $P$ when $\alpha=1$. The variance difference $f_t>1$ still holds there. Weaken to "in general" or state the condition ($\alpha<1$ in this example).
- E6 never mentions the assumptions C30, C33, C34, C35 that it relies on through C4 (it does cite C4).

---

## C6 (E9): exact likelihood and scale equation for log-normal ADAM ETS

**Outcome: confirmed, full.**

Derivation. By C12, $\log\mathbf v_t=\mathbf F\log\mathbf v_{t-\mathbf l}+\mathbf g u_t$, $\log z_t=\mathbf w'\log\mathbf v_{t-\mathbf l}+u_t$ with $u_t=-s/2+\eta_t$, $\eta_t\sim\mathcal N(0,s)$ (C36), independent (C30), initial states fixed (C34). This is the linear Gaussian SSOE model with a known mean shift, so the D8 filter with $\mathbf a_{t+1}=\mathbf F\mathbf a_t-\mathbf g s/2$ (unobserved) and $\mathbf a_{t+1}=\mathbf F\mathbf a_t-\mathbf g s/2+\mathbf k_tv_t$ (observed), $v_t=\log z_t-\mathbf w'\mathbf a_t+s/2$, gives $\log z_t\mid Y_{t-1}\sim\mathcal N(\mathbf w'\mathbf a_t-s/2,sf_t)$ (C4); $\mathbf P_t,f_t,\mathbf k_t$ do not depend on $s$. The density of $z_t$ adds the Jacobian $-\log z_t$. So
$$\ell(\theta,s)=-\tfrac12\sum_{O_t=1}\Big[\log(2\pi sf_t)+\frac{v_t(s)^2}{sf_t}\Big]-\sum_{O_t=1}\log z_t,\quad v_t(s)=v_t^0+sb_t$$
(the recursion is affine in $s$ with $s$-free coefficients). Differentiating,
$$\partial_s\ell=-\frac{T_1}{2s}+\sum\frac{v_t^2}{2s^2f_t}-\sum\frac{v_tb_t}{sf_t};$$
multiplying by $2s^2$ and substituting $v_t=v_t^0+sb_t$, the cross terms $\pm2s\sum v^0b/f$ cancel and
$$-T_1s+\sum\frac{(v^0)^2}{f_t}-s^2\sum\frac{b_t^2}{f_t}=0\iff\Big(\sum\frac{b_t^2}{f_t}\Big)s^2+T_1s-\sum\frac{(v_t^0)^2}{f_t}=0,$$
as claimed. The quadratic has product of roots $<0$ (when $\sum b^2/f>0$), so exactly one positive root.

Numerical checks (`chk2.py`).
- Exactness: ETS(M,N,N) in logs, $\alpha=0.35$, $s=0.4$, 12 periods with periods 3 and 7 unobserved; brute-force integral of the complete-data density over the two unobserved log-sizes ($801^2$ grid) gave log-likelihood $-18.27559945773717$; the filter gave $-18.27559945773701$.
- Scale: with $v_t^0$ and $b_t$ extracted from the filter (linearity in $s$ confirmed: $v(0.7)=v^0+0.7\,b$ to machine precision), the positive root of the quadratic was $0.29147142573$ (4 gaps pattern) and $0.25189682446$ (2 gaps), identical to the fine grid argmax of the filter likelihood ($0.2914714257$, $0.2518968245$). Coefficients $b_t\in[0.5,0.85]$.

Gaps and typos.
- The claim says "the known error mean $-s/2$ added to the propagation"; the state propagation needs $-\mathbf gs/2$ (as the manuscript has it). Wording.
- E9.3 asserts the linearity and the form of $b_t$ without giving $b_t$; E9.4 does not show the cancellation of cross terms. Both are routine but should be written. The positive-root statement (a quadratic with $\sum b^2/f>0$ and $\sum v_0^2/f>0$) is not argued in E9.
- E9 cites C12, C36, C4 but not C33 (missing at random) and C34 (fixed initial states), which are in the claim's assumptions and are used.

---

## C8 (E10): ADAM's log-normal scale estimator

**Outcome: confirmed, full** (as a statement about the stated formula; that ADAM's code uses this estimator is a fact about software that I did not check).

Derivation. All periods observed: $f_t=1$, $\mathbf k_t=\mathbf g$ (C15), and $v_t(s)=u_t+s/2$ with $u_t=\log(1+\epsilon_t)$ the realised log error, so $v^0_t=u_t$, $b_t=1/2$. The C6 equation gives $\tfrac{T_1}{4}s^2+T_1s-T_1m=0$, i.e. $s^2+4s-4m=0$ with positive root $2(\sqrt{1+m}-1)$. (Independent derivation from the likelihood $-\tfrac T2\log s-\tfrac{1}{2s}\sum(u+s/2)^2$: the same quadratic.) The ADAM-type estimator solves $s^2-4s+4m=0$, roots $2\pm2\sqrt{1-m}$; the smaller is $2(1-\sqrt{1-m})$, real only for $m\le1$. Since $\sqrt{1+m}+\sqrt{1-m}<2$ for $0<m\le1$ (concavity), $2(1-\sqrt{1-m})>2(\sqrt{1+m}-1)$ for all such $m$, so it overestimates for every $m$. At $s=0.5$: $m=s+s^2/4=0.5625$ (since $\mathrm E\,u^2=\mathrm{Var}\,u+(\mathrm Eu)^2=s+s^2/4$), MLE $=2(1.25-1)=0.5$ exactly, ADAM-type $=2(1-0.6614)=0.677$, ratio $1.354$.

Numerical checks (`chk2.py`): at $m=0.5625$ the ADAM-type value is $0.6771243$, the MLE $0.5$, ratio $1.3542$. Simulation of $2\times10^5$ log-normal errors with $s=0.5$: sample $m=0.56123$, grid argmax of the log-normal likelihood $0.49899$, formula $2(\sqrt{1+m}-1)=0.49899$, ADAM-type $0.6752$. On a grid of 999 values of $m$ in $(0,1)$ the ADAM-type value exceeds the MLE at every one.

Gaps and typos.
- E10.1 cites C15, which is not in C8's `depends_on` (only C7, via which C6 enters). Add C15 and C6 or cite C7.
- "Undefined for $m>1$" is true (real root requires $m\le1$) but is not stated in E10.
- E10.3 evaluates the estimators at $\mathrm Em$ (plug-in), which is what the claim says ("when $m=s+s^2/4$"); the sample-level behaviour differs slightly (0.675 against 0.677 in the simulation above) and the proof of "overestimates for every $m$" is not in E10.
- "used in ADAM" is not supported by evidence in the claimset (no cited code location).

---

## C10 (E12): prediction interval of demand that counts zeros as covered

**Outcome: confirmed, partial.** The coverage identity is verified; "centred on the size forecast rather than on the demand forecast" is an interpretation that E12 does not derive.

Derivation. With $y=oz$, $z>0$, $o\perp z$: for $u\ge0$, $P(0\le y\le u)=P(o=0)+P(o=1,z\le u)=(1-p)+pF_z(u)$. Setting this equal to $c$ gives $F_z(u)=(c-(1-p))/p$. Hence the upper bound is the $(c-(1-p))/p$ quantile of the size distribution, whose location is the size forecast $\hat\mu_t$ (the demand forecast $p_t\hat\mu_t=\mathrm E\,y_t$ is the mean of a mixture, not a location of $z$). An interval with positive lower bound does not cover zero, so an interval that counts zeros as covered has the form $[0,u]$ and the identity is the complete description.

Numerical check (`chk2.py`): $p=0.4$, $c=0.9$, log-normal sizes (median 3, $\sigma=0.5$), $u=F_z^{-1}(0.75)$; $2\times10^6$ draws of $y=oz$: $P(y\le u)=0.89972$ (target 0.9), $P(z\le u)=0.75000$ (target 0.75).

Gaps and typos.
- Needs $c>1-p_t$ (otherwise the required size coverage is negative; the interval is $\{0\}$ with coverage $1-p_t\ge c$). The claim and E12 should say so.
- E12.1 needs $z\ge0$ (positive sizes), not only continuity (C32), and the interval to be of the form $[0,u]$.
- "Centred on $\hat\mu_t$": E12.2 only says "a quantile of the sizes around $\hat\mu_t$". The statement that the bounds are quantiles of $F_z$, a family located at $\hat\mu_t$, is true but is not what "centred" literally says (a quantile at level 0.75 is above the location); rephrase as "bounds are quantiles of the size distribution".

---

## C12 (E8): logarithms of ADAM and conventional multiplicative ETS

**Outcome: confirmed, full.**

Derivation. ADAM ETS(M,Md,M): $\log z_t=\log l_{t-1}+\phi\log b_{t-1}+\log s_{t-m}+u_t$; $\log l_t=\log l_{t-1}+\phi\log b_{t-1}+\alpha u_t$; $\log b_t=\phi\log b_{t-1}+\beta u_t$; $\log s_t=\log s_{t-m}+\gamma u_t$. These are the equations of ETS(A,Ad,A) with $\epsilon_t$ replaced by $u_t$, so $\mathbf w,\mathbf F,\mathbf g$ are those of the additive model: the SSOE form D1 holds in the logs. Conventional form: the measurement is the same, but $\log l_t=\log l_{t-1}+\phi\log b_{t-1}+\log(1+\alpha\epsilon_t)$ and so on, and $\log(1+g\epsilon)=g\log(1+\epsilon)$ for all $\epsilon$ only if $g\in\{0,1\}$ (the function $\epsilon\mapsto\log(1+g\epsilon)/\log(1+\epsilon)$ is not constant for $0<g<1$), so the innovations are not proportional to $u_t$.

Numerical checks (`chk2.py`). Simulated ETS(M,Md,M), $m=4$, 40 periods with random $1+\epsilon$: the logs of level, trend, the four seasonal states and the observation errors matched the additive-form log recursion with innovation $u_t$ (max difference $1.8\times10^{-15}$). For $g=0.3$ and $\epsilon\in\{-0.5,-0.2,0.3,1.0\}$ the ratio $\log(1+g\epsilon)/(g\log(1+\epsilon))$ was $0.782,0.924,1.095,1.262$, not constant.

Gaps and typos.
- E8 omits the seasonal update (step 4) and does not state that the matrices coincide with the additive model's; it also gives no step for "not proportional" (add: the ratio above is non-constant for $0<g_k<1$). The claim is true for $0<g_k<1$ only (for $g_k\in\{0,1\}$ the innovations coincide).
- In the conventional form the claim requires $1+g_k\epsilon>0$ for the logarithm to exist (a condition on the error support); not stated.

---

## C14 (E15): sign of the entropy term

**Outcome: confirmed, partial.** The identity $\int f\log f=-H$ and the resulting sign are verified; whether Svetunkov and Boylan (2023) print the opposite sign was not verified, and E15's key premise is circular.

Derivation. $H=-\mathrm E\log f=-\int f\log f$, so $\int f\log f=-H$. If the term the objective adds at each zero is the expected log-density $\int f\log f$ (the E-step analogy), it equals $-H_z$, and the entropy enters with a negative sign.

Numerical checks (`chk2.py`). $\int f\log f$ on a fine grid against $-H$ in closed form: Gamma(shape 2, scale 0.5): $-0.88406848$ against $-0.88406848$; Gamma(5, 0.2): $-0.54414524$ against $-0.54414525$; Normal($\sigma=0.7$): $-1.06226359$ against $-1.06226359$.

Gaps and typos.
- E15.3 asserts that the term added is $\int f_z\log f_z$ and cites D4 as "definition of the objective", but D4 itself defines the objective with "minus the differential entropy". That makes the premise equivalent to the conclusion; the claim that the published sign is wrong cannot follow from D4. The premise should be the motivation (the expected complete-data log-density of the missing size in the E-step), with the sign of the published equations given as the thing being corrected.
- D4 is stated as the objective "of Svetunkov and Boylan (2023, eq. B.6)" with a minus sign, while C14 says that paper has the other sign. One of these misdescribes the paper; D4 should say "as corrected here" or describe the printed form.
- E15 is only about the sign convention; it does not by itself justify that $\int f\log f$ is the right additive term (that is C2's topic and the stationary points differ, so the term is not "right" in the EM sense).
- Typo: "the term added ... (the E-step analogy ...)" is written as a step, not a derivation.

---

## C16 (E7): gain after an interval for ETS(A,N,N)

**Outcome: confirmed, full.**

Derivation. D8 with $w=F=1$, $g=\alpha$, started at $P=0$. Unobserved: $P_{t+1}=P_t+\alpha^2$, so $P=(j-1)\alpha^2$ after $j-1$ unobserved periods, $f=P+1$, $k=(P+\alpha)/f=((j-1)\alpha^2+\alpha)/((j-1)\alpha^2+1)$. $j=1$ gives $\alpha$. $k\to1$ as $j\to\infty$ if $\alpha\neq0$.
The premise $P=0$ before the interval is consistent: after an observed period with $P=0$, $P_{t+1}=\alpha^2-\alpha^2=0$ (C15), so the same formula holds for every interval.

Numerical checks (`chk1.py`, `chk2.py`). The recursion D8 run on gaps of $j-1=0..6$ unobserved periods followed by one observed, for $\alpha\in\{0.1,0.3,0.7\}$: the gain equals the formula to $10^{-14}$. For $\alpha=0.3$, $j=200$: $0.96298$. Independent Monte Carlo ($2\times10^6$ paths, $\alpha=0.3$, $j=4$): regression coefficient of the new level on the observation (the conditional-mean update, the gain) $0.44867$ against formula $0.44882$.

Gaps and typos.
- E7.1 does not say that $P=0$ at the start (it is an assumption of the claim), nor does it show $P$ grows by $\alpha^2$ per unobserved period (this is the D8 propagation).
- "Tends to one" needs $\alpha\neq0$ (for $\alpha=0$ the gain is 0 for every $j$).
- E7 does not mention C4 (the claim lists it in `depends_on`), although D8 is the object.

---

## C19 (E13): zero-error forecast is not the conditional mean for ADAM multiplicative ETS

**Outcome: confirmed, partial.** The ETS(M,N,N) Gamma formula and the Jensen step are verified. The first sentence is stated for the whole ADAM multiplicative family but is proved only for ETS(M,N,N), and it needs $h\ge2$.

Derivation. ETS(M,N,N): $z_{T+h}=l_{T+h-1}(1+\epsilon_{T+h})$, $l_{T+h-1}=l_T\prod_{i=1}^{h-1}(1+\epsilon_{T+i})^\alpha$, and the future errors are independent of each other and of $l_T$ (C30), so
$$\mathrm E(z_{T+h}\mid Y_T)=l_T\,[\mathrm E(1+\epsilon)^\alpha]^{h-1}\,\mathrm E(1+\epsilon)=\hat\mu_{T+h}[\mathrm E(1+\epsilon)^\alpha]^{h-1}$$
with $\hat\mu_{T+h}=l_T$. Jensen with $x\mapsto x^\alpha$ strictly concave on $(0,\infty)$ for $0<\alpha<1$ gives $\mathrm E(1+\epsilon)^\alpha<1$ when $1+\epsilon$ is not degenerate. For Gamma with shape $1/s$ and scale $s$ (mean 1, variance $s$): $\mathrm EX^\alpha=s^\alpha\Gamma(1/s+\alpha)/\Gamma(1/s)$.

Numerical checks (`chk2.py`). $s=0.32$, $\alpha=0.21$: formula $0.973088887$; Monte Carlo $(4\times10^6)$ $0.973059$. Conditional mean ratio $\mathrm E(z_{T+h}\mid Y_T)/\hat\mu_{T+h}$ from $10^6$ simulated paths against $0.973089^{h-1}$: $h=1$: $0.9995$ vs $1$; $h=2$: $0.97255$ vs $0.97309$; $h=5$: $0.89632$ vs $0.89662$; $h=12$: $0.74099$ vs $0.74076$ (all within Monte Carlo error).

Gaps and typos.
- At $h=1$ the zero-error forecast is the conditional mean (the formula gives 1); the claim should say $h\ge2$ and non-degenerate errors ($s>0$, $0<g_k<1$).
- For ETS with trend or seasonality the same $\epsilon$ drives several components, so the exponents add (e.g. $(1+\epsilon)^{\alpha+\beta+\dots}$ in a product of components); the effective exponent can exceed 1, and then $\mathrm E(1+\epsilon)^c>1$. The sentence "because $\mathrm E(1+\epsilon_t)^{g_k}<1$ for $0<g_k<1$" is therefore not a proof for ETS(M,Md,M), only for models whose forecast is driven by a single component. Restrict the first sentence to ETS(M,N,N) or add the argument.
- E13.2 uses $\mathrm E(1+\epsilon_{T+h})=1$ without stating it; E13.3 cites E13.2 as its premise, but it is a moment computation of the Gamma law (no dependence on E13.2).
