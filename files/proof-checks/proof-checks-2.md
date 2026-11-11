# Independent proof check, round 2: revised claims C2, C5, C10, C14, C19

Checked by an AI agent (Claude Sonnet 5.5 in Claude Code), not an author, on 2026-11-11. This is a second pass on five claims (and their evidence E2, E6, E12, E15, E13) that the authors revised after my first report (`proof-checks.md`, unchanged). Method: I re-read each claim as it now stands in the claimset, with the definitions it marks (D4 now defines the term of each zero as $\int f_z\log f_z\,dz$ without a sign), re-derived it by hand from its premises, checked every evidence step against the claim, and spot-checked identities with Python/numpy (scipy not installed). Numbers come from `chk3.py` (round 1, re-used for C2: normal sizes with fixed location), `chk2.py` (round 1, re-used for C10, C14, C19) and the new `chk4.py` (C5). I did not consult Svetunkov and Boylan (2023).

| Claim | Evidence | Outcome | Extent |
|---|---|---|---|
| C2 | E2 | confirmed | full |
| C5 | E6 | confirmed | full |
| C10 | E12 | confirmed | full |
| C14 | E15 | confirmed | full |
| C19 | E13 | confirmed | full |

Smaller changes (do they leave my round-1 full confirmations valid?):
- C1 ("positive continuous size"): yes; the positivity I asked for is now a premise and E1.1/E1.2 use it, nothing else changed.
- C6 (mean $-s/2$ in the prediction, multiplied by $\mathbf g$ in the propagation; E9.2 cites C33, C34): yes; this is exactly the recursion I used in the derivation and the brute-force check (propagation $-\mathbf gs/2$), and C33/C34 are the assumptions it needs.
- C8 (depends on C15): yes; C15 is the source of $f_t=1$, $\mathbf k_t=\mathbf g$ in E10.1, which closes the missing dependency.
- C16 ("when $\alpha\neq0$", E7.1 cites C4): yes; the limit statement is now correct for $\alpha\neq0$ and the formula derivation is unchanged.

---

## C2 (E2): stationary points of the entropy-augmented likelihood

**Outcome: confirmed, full.**

The claim no longer asserts that the E-step is a cross-entropy in the dynamic model; it asserts three things.

1. At the current estimate the gradient of $Q(\cdot\mid\theta^{(k)})$ equals the score of $\ell$ (E2.1, Fisher's identity). Derivation: $\nabla\log p(y;\theta)=\nabla p(y;\theta)/p(y;\theta)=\int p(y,z;\theta)\nabla\log p(y,z;\theta)\,dz/p(y;\theta)=\mathrm E_{z\mid y,\theta}\nabla\log p(y,z;\theta)$, differentiation under the integral being the regularity condition. This holds at every $\theta^{(k)}$, hence at an EM fixed point, as the claim says; E2.2 follows. It holds for the dynamic model too, since it needs no form of the posterior.
2. The term for a zero, $-H_z(\theta)=\int f_\theta\log f_\theta$, has non-zero gradient in general: for the normal, $-H_z=-\tfrac12\log(2\pi e\sigma^2)$ with derivative $-1/(2\sigma^2)$ (E2.4).
3. For normal sizes with fixed location: $\ell(\sigma^2)=-\tfrac{T_1}2\log\sigma^2-\tfrac S{2\sigma^2}+c$, stationary at $S/T_1$ (E2.3); $\ell_H=\ell-\tfrac{T_0}2\log(2\pi e\sigma^2)$ has derivative $-\tfrac{T_1}{2\sigma^2}+\tfrac S{2\sigma^4}-\tfrac{T_0}{2\sigma^2}$, zero at $S/T$, $T=T_1+T_0$ (E2.4). $S/T\neq S/T_1$ iff $T_0>0$ and $S>0$ (E2.5). With fixed location the zero-error and exact one-step densities coincide, so $\ell$ and the size part of $\ell_H$ are the same function; E2.4's use of $\ell$ is right.

Numerical check (`chk3.py`, $T_1=60$, $T_0=40$, normal $e_t$ with $S=\sum e_t^2$): argmax of $\ell_H$ $=1.9910$ against $S/T=1.99084$; argmax of $\ell$ and the EM iteration limit $3.3181$ against $S/T_1=3.31806$. Cross-entropy $\int N(0,1.3)\log N(0,\sigma^2)$ has derivative $0.0$ at $\sigma^2=1.3$; the derivative of $-H$ there is $-0.38462=-1/(2\cdot1.3)$.

Remaining gaps and typos.
- E2.1 is a cited standard identity (B1) rather than derived in the steps; I verified the derivation above and, for the static normal, numerically.
- The claim says "stationary at different variances" for the example only; "differ in general" rests on that example plus the non-zero gradient. This is adequate for "in general" (a single case where they differ), but the claim does not say that the objectives differ also in the dynamic models (where $\ell_H$ uses zero-error densities as well, C5).

---

## C5 (E6): zero-error recursion likelihood versus the observed-data likelihood

**Outcome: confirmed, full** (for the claim as now worded: "in general not", with the condition on $c_i$).

Derivation. From D8 started at $\mathbf P=0$ (C34), after $j-1$ unobserved periods $\mathbf P=\sum_{i<j}\mathbf F^{i-1}\mathbf g\mathbf g'\mathbf F'^{\,i-1}$ and $f=1+\mathbf w'\mathbf P\mathbf w=1+\sum_{i<j}c_i^2>1$ if some $c_i\neq0$ (E6.1, E6.2 correct). By C4 the exact log-likelihood has variance $\sigma^2f_t$ at that period; the zero-error recursion D5 uses $\sigma^2$, so its log-likelihood $-\tfrac12\sum[\log(2\pi\sigma^2)+e_t^2/\sigma^2]$ is a different function from $-\tfrac12\sum[\log(2\pi\sigma^2f_t)+v_t^2/(\sigma^2f_t)]$ (E6.3): not the observed-data likelihood. Maximising over $\sigma^2$ for given other parameters gives $\hat\sigma^2_e=\tfrac1{T_1}\sum e_t^2$ and $\hat\sigma^2_v=\tfrac1{T_1}\sum v_t^2/f_t$ (E6.5). They are equal iff $Q=\sum e_t^2-\sum v_t^2/f_t=0$. At the first observed period after the first interval both recursions predict from $\mathbf F^{j-1}\mathbf x_0$, so $e_t=v_t$ and the term contributes $e_t^2(1-1/f_t)>0$. $Q$ is therefore not identically zero: take a sample in which that period is the last observation; then earlier observed periods have $\mathbf P=0$, $f=1$, $e=v$ and $Q=e_t^2(1-1/f_t)>0$ for any $e_t\neq0$.

Numerical checks (`chk4.py`, ETS(A,N,N), $\alpha=0.4$, $l_0=0$). A 12-period sample with periods 3, 4 and 7 unobserved: $\sum v_t^2/f_t=6.7015$, $\sum e_t^2=7.4975$; at the first observed period after the gap $v_t=e_t=1.2446$ and $f_t=1.32$, while at the following period $v_t=0.0383\neq e_t=0.2194$. For a 5-period sample ending with an observation after a gap: $Q=0.124228$ against $e_t^2(1-1/f_t)=0.124228$. Over 2000 random 12-period samples $Q$ was positive in all of them (minimum $2.4\times10^{-4}$, maximum 36.7) and nonzero in 100% of draws. Round-1 simulation (`chk1.py`): exact MLE of $\alpha$ averaged 0.301 against the zero-error maximiser 0.397 for true 0.3.

Remaining gaps and typos.
- E6.4 says that both predict "the first observed period after an interval" from the same propagated states, so that $e_t=v_t$ there. That is true after the first interval (from $\mathbf x_0$) and, more generally, when the states at the start of the interval coincide in the two recursions; after an earlier gap the states differ through the gain ($v_t\neq e_t$ at the second observed period, shown above). The step should say "after the first interval" (that is all the argument needs).
- E6.5's "unless the data satisfy $\sum e^2=\sum v^2/f$" is a true iff; the non-vanishing of $Q$ on a set of data (as above) is left implicit.
- The claim concerns the maximiser in $\sigma^2$ at given other parameters; the joint maximiser over the other parameters also differs (round-1 simulation) but this is not part of the claim.

---

## C10 (E12): prediction interval of demand that counts zeros as covered

**Outcome: confirmed, full.**

Derivation. $y=oz$, $z>0$, $o\perp z$. For $u\ge0$: $P(0\le y\le u)=P(o=0)+P(o=1,\,z\le u)=(1-p)+pF_z(u)$ (E12.1). Setting it to $c$ with $1-p<c\le1$: $F_z(u)=(c-(1-p))/p\in(0,1]$, so $u$ is the quantile of the size at that level (E12.2; for $c=1$ and unbounded sizes $u$ is infinite, a trivial edge). The size $z$ has mean $\hat\mu_t$ (the size forecast) and the demand $\mathrm E(o_tz_t)=p_t\mathrm E z_t=p_t\hat\mu_t$ by independence (E12.3), so the quantile belongs to a distribution with mean $\hat\mu_t$, not $p_t\hat\mu_t$.

Numerical check (`chk2.py`): $p=0.4$, $c=0.9$, log-normal sizes; level $(0.9-0.6)/0.4=0.75$; with $2\times10^6$ draws $P(y\le u)=0.89972$ (target 0.9) and $P(z\le u)=0.75000$.

Remaining gaps and typos.
- "The size has mean $\hat\mu_t$" (E12.3) is a premise about what $\hat\mu_t$ means, not derived; it holds when $\hat\mu_t$ is defined as the conditional mean of the size. For the ADAM multiplicative ETS the zero-error forecast is not the conditional mean at $h\ge2$ (C19), so there the size forecast in the claim must be read as the conditional mean, or the sentence as "location" of the size distribution.
- E12.3 cites C31 only; the "mean" premise itself has no support in the steps.

---

## C14 (E15): sign of the entropy term

**Outcome: confirmed, full** (for the claim as now stated, which is the identity and not a statement about the printed sign in the 2023 paper).

Derivation. $H_z=-\mathrm E\log f_z=-\int f_z\log f_z\,dz$ (D32), so $\int f_z\log f_z\,dz=-H_z$; the term of each zero in D4 is $\int f_z\log f_z\,dz$, hence it equals $-H_z$ and the entropy enters with a negative sign. D4 now defines the term without a sign, so E15.3 is a restatement of D4 and the argument is not circular any more.

Numerical checks (`chk2.py`): $\int f\log f$ on a fine grid against $-H$ in closed form: Gamma(2, 0.5) $-0.88406848$ vs $-0.88406848$; Gamma(5, 0.2) $-0.54414524$ vs $-0.54414525$; Normal($\sigma=0.7$) $-1.06226359$ vs $-1.06226359$.

Remaining gaps and typos.
- Whether D4 faithfully describes eq. B.6 of Svetunkov and Boylan (2023), and what sign the paper prints, is not checked by this proof or by me; the claim now says only what the term equals.
- E15.2 "multiplying by $-1$" is trivial; fine.

---

## C19 (E13): zero-error forecast and conditional mean for ETS(M,N,N)

**Outcome: confirmed, full.**

Derivation. ADAM form: $z_{T+h}=l_{T+h-1}(1+\epsilon_{T+h})$, $l_{T+h-1}=l_T\prod_{i=1}^{h-1}(1+\epsilon_{T+i})^\alpha$. With independent errors of mean one, $\mathrm E(z_{T+h}\mid Y_T)=l_T[\mathrm E(1+\epsilon)^\alpha]^{h-1}\cdot1=\hat\mu_{T+h}[\mathrm E(1+\epsilon)^\alpha]^{h-1}$ with $\hat\mu_{T+h}=l_T$ (E13.2). For $0<\alpha<1$, $x^\alpha$ is strictly concave on $(0,\infty)$, so for non-degenerate $1+\epsilon>0$ of mean one, $\mathrm E(1+\epsilon)^\alpha<(\mathrm E(1+\epsilon))^\alpha=1$ (E13.1); the factor is $<1$ to the power $h-1\ge1$ for $h\ge2$ (E13.3). Gamma shape $1/s$, scale $s$: $\mathrm EX^\alpha=s^\alpha\Gamma(1/s+\alpha)/\Gamma(1/s)$ (E13.4). The restriction to ETS(M,N,N), $0<\alpha<1$ and $h\ge2$ fixes the two overreaches of round 1.

Numerical checks (`chk2.py`). $s=0.32$, $\alpha=0.21$: formula $0.973088887$, Monte Carlo ($4\times10^6$) $0.973059$. Ratio of simulated conditional mean to $\hat\mu_{T+h}$ against $0.973089^{h-1}$: $h=2$: $0.97255$ vs $0.97309$; $h=5$: $0.89632$ vs $0.89662$; $h=12$: $0.74099$ vs $0.74076$ ($10^6$ paths, within Monte Carlo error); $h=1$: $0.9995$ vs $1$.

Remaining gaps and typos.
- The claim marks `[[D20:ETS(M,N,N)]]`, but D20 is the simulation setting with $s=0.32$, $\alpha=0.21$, 100 observations and initial level 10 (kind `data`), not the model class ETS(M,N,N); the claim is general. The marker should point to a definition of ETS(M,N,N) (D12, which defines the iETS(M,N,N) form, or D6) or be dropped.
- E13.4 cites E13.2 as its premise, but the Gamma moment needs only the distribution; harmless.
- The independence of the future errors from $l_T$ (given $Y_T$) is used in E13.2 and covered by C30; fine.
