QFLARE / EXPERIMENTAL REPORT

13 November 2026

# QFLARE: tested attack strategies

Brief mathematical foundations, recorded results, and best-method pseudocode

The best recorded result is **1.50005821**, obtained by **clientwise Stall + orthogonal updates with dynamic timing**. This report summarizes the supplied formulations and code; it does not introduce a new attack or rerun the benchmark. Screenshot row numbers link each method to the results on page 4.

## 1. Common setup and interpretation

Eight clients train a quantum classifier for handwritten digits 3 and 8. There are 60 rounds, $t=0,\ldots,59$. Only clients $i\in\{7,8\}$ are controlled. Let $q_t\in\mathbb R^{24}$ be the shared circuit parameters, $d_{i,t}$ the honest circuit update, and $a_{i,t}$ its submitted replacement. The ten classical-head coordinates remain honest. Define 

$$
m_{i,t}=\left\lVert d_{i,t}\right\rVert_2,\qquad R_{i,t}=2\max(m_{i,t},0.01),\qquad
 \left\lVert a_{i,t}-d_{i,t}\right\rVert_2\le R_{i,t}.
$$

 At most 30 rounds may change; changing both clients counts once. On a skipped round, $a_{i,t}=d_{i,t}$. A candidate $x$ is made feasible by 

$$
\Pi_{i,t}(x)=d_{i,t}+\min\!\left(1,\frac{R_{i,t}}{\left\lVert x-d_{i,t}\right\rVert_2}\right)(x-d_{i,t}),
 \quad \Pi_{i,t}(d_{i,t})=d_{i,t}.
$$

 The recorded score is $S=\frac{100}{6}\sum_{k=1}^{6}(F1_{\mathrm{clean},k}-F1_{\mathrm{attack},k})$: the average macro-F1 drop over the six evaluation cases, in percentage points. Larger is better for the attack; negative values mean higher attacked F1 on average. Thus $1.50005821$ corresponds to a mean F1 difference of $0.0150005821$, not a relative 1.5% reduction. \[1,2\]

## 2. Timing rules: when the payload is applied

**Inverse log.** With a fixed seed 2026, draw $B\sim U(2,10)$ once and select 

$$
\mathcal A=\left\{1+k+\left\lfloor29\frac{B^{k/29}-1}{B-1}\right\rfloor:k=0,\ldots,29\right\}.
$$

 This gives 30 distinct opportunities from round 1 to 59, denser early in training. It is the inverse of a shifted logarithm. No-op slots are not rescheduled. \[3,4,7\]

**Dynamic timing.** Let $M_t=(m_{7,t}+m_{8,t})/2$, $N_t$ be the changed-round count, and initialize $\tau=0.025$ (or the mean of supplied clean calibration magnitudes, floored at $10^{-12}$). Subject to valid required history, attack if 

$$
1\le t<60,\quad N_t<30,\quad
 \bigl(M_t\ge\tau_t\ \text{or}\ 30-N_t\ge60-t\bigr).
$$

 After the decision, retain the latest three $M$ values, including skips. Sort them as $x_1\le x_2\le x_3$ and update 

$$
\tau_{t+1}=\max\left(10^{-12},
 \begin{cases}
 x_2,&\text{at least two }x_j>\tau_t,\\
 (x_2+x_3)/2,&x_3<\tau_t,\\
 \tau_t,&\text{otherwise}.
 \end{cases}\right)
$$

 With fewer than three observations, keep the threshold. This selects large-update opportunities and forces use of the remaining budget when necessary. \[6,8,2\]

**Other schedules.** Shared uniform Stall uses $1,3,\ldots,59$; custom original Stall uses $2,4,\ldots,58$ (29 slots). Max-Stall ablations use $1,\ldots,30$, or $1,\ldots,15$ followed by $17,19,\ldots,45$. The last sequence clarifies the abbreviated screenshot label in row 21. \[2,5\]

## 3. Baselines and history-based methods

In formulas on this page, current-round subscripts are suppressed when unambiguous; $\operatorname{unit}(x)=x/\left\lVert x\right\rVert_2$ for $x\ne0$.

**Noise (rows 2, 8).** Randomly change direction, magnitude, or both. For $m_i>0$, select a perpendicular unit vector $w$ and set $v=\cos\psi\,d_i/m_i+\sin\psi\,w$. Direction changes use $\psi\sim U(0,\pi)$; otherwise $\psi=0$. Submit 

$$
a_i=\rho v,\qquad \rho_{\max}=m_i\cos\psi+\sqrt{R_i^2-m_i^2\sin^2\psi},
$$

 where $\rho=m_i$ for direction-only noise and $\rho\sim U(0,\rho_{\max})$ otherwise. Near zero, use a random unit $v$ and target $d_i+(R_i/2)v$. Row 2 uses inverse-log timing. For row 8, the conversation records six randomized intervals with quotas totaling 30; the interval boundaries and quotas are not included in the supplied files, so they are not reconstructed. \[3,2 p.4\]

**Original Stall (rows 3, 6, 23).** Submit $a_{i,t}=\Pi_{i,t}(-d_{i,t-1})$: oppose each client’s previous *honest* proposal. This does not undo the global update. The three rows use inverse-log, alternating custom, and dynamic timing, respectively. \[3,5,2 pp.56–61\]

**Echo (rows 4, 7).** Replay the previous honest update: 

$$
a_{i,t}=\Pi_{i,t}\bigl((1-\alpha_t)d_{i,t}+\alpha_t d_{i,t-1}\bigr).
$$

 Let $s=\cos(d_7,d_8)$, $c=\cos(\bar d_t,\bar d_{t-1})$, $h=|M_t-M_{t-1}|/\max(M_t,M_{t-1},10^{-12})$, and $\bar d_t=(d_{7,t}+d_{8,t})/2$. The strict gate is $s\ge.5$, $M_t\ge.025$, $c\le.5$, $h\ge.25$. Use $\alpha=1$ if strict, otherwise $1/2$. Inverse-log Echo applies this payload on scheduled rounds. Custom Echo requires the gate with relaxation $r=0,.25,.50$ in rounds 1–9, 10–34, 35–59: multiply the lower thresholds by $1-r$ and the upper threshold on $c$ by $1+r$; the middle phase also requires a fair coin. Undefined cosines skip custom Echo. \[3,9\]

**Casewise (row 5).** Draw thresholds uniformly from $\{0,.25,.50\}$ for direction agreement $D$, $\{.10,.25,.50\}$ for magnitude similarity $E$, and $\{.01,.025,.05\}$ for large mean size $L$. Test $s\ge\tau_D$, $|m_7-m_8|/\max(m_7,m_8)\le\tau_E$, and $M\ge\tau_L$. \[4\]

| Condition         | Candidate action on an inverse-log opportunity                              |
|:------------------|:----------------------------------------------------------------------------|
| $\neg D,E,L$      | Halve the smaller update; set the other to zero (history breaks ties).      |
| $D,\neg E,L$      | Fair choice: zero both or reverse both, $a_i=-d_i$.                         |
| $\neg D,E,\neg L$ | Random client: $a_i=-(R_i-m_i)d_i/m_i$; zero the other.                     |
| $D,\neg L$        | Uniform choice of the three reversal actions below, shared by both clients. |
| $D,E,L$           | Set both updates to zero.                                                   |
| $\neg D,\neg E$   | Keep both honest.                                                           |

For $u_i=d_i/m_i$ and $v=\operatorname{unit}(q_t-q_{t-1})$, the three actions are $-(R_i-m_i)u_i$, $d_i-R_iv$, and $-\lambda_i v$, where $\lambda_i=-d_i^\top v+\sqrt{(d_i^\top v)^2+R_i^2-m_i^2}$. Missing history selects the first; near-zero honest vectors skip.

**Shared Stall (rows 9–10).** Oppose the average of the last three observed global movements and submit an identical vector to both clients, within both radius bounds. The conversation confirms this construction and the inverse-log/uniform comparison, but does not expose its exact scalar-selection code. \[2 p.10\]

**Current cancellation (row 11).** Submit $a_7=a_8=0$ on accepted coherent-update cases. Fixed gates reject negligible, conflicting, unbalanced, or oscillating updates and require coordinate-sign agreement covering at least 80% of update magnitude. Remaining exact gate constants are absent. \[2 pp.11–12\]

## 4. Orthogonal, maximum-Stall, and mixed methods

**Orthogonal foundation.** For observed movement $g_r=q_{r+1}-q_r$, define corrected history 

$$
H_r=g_r-\widetilde w_{7,r}a_{7,r}-\widetilde w_{8,r}a_{8,r}.
$$

 The default weights are $1/8$ each. This subtracts the controlled clients exactly only for linear FedAvg with correct weights; for median/trimmed mean it is a proxy. Form $Z=[d_{7,t},d_{8,t},H_{t-3},H_{t-2},H_{t-1}]$ using available history. With an orthonormal basis $Q$ of its nonzero columns, let $P=I-QQ^\top$ and choose a unit $v$ in its nullspace. Then 

$$
a_{i,t}=\sqrt{R_{i,t}^2-m_{i,t}^2}\,v,
 \qquad \left\lVert a_{i,t}-d_{i,t}\right\rVert_2^2=m_{i,t}^2+(R_{i,t}^2-m_{i,t}^2)=R_{i,t}^2.
$$

 The resultant is perpendicular to both current honest updates and retained history; the perturbation cancels the honest component. The stable default is $v=Pe_j/\left\lVert Pe_j\right\rVert_2$ for the largest projected coordinate norm. \[6,7\]

**Coordinated orthogonal + reversal (row 12; likely source match).** If $\cos(d_7,d_8)\ge.5$, use $w=\operatorname{unit}(d_7/m_7+d_8/m_8)$ and 

$$
a_i=-\lambda_i w,\qquad \lambda_i=-d_i^\top w+\sqrt{(d_i^\top w)^2+R_i^2-m_i^2}.
$$

 Otherwise use orthogonal resultants. Dynamic timing controls both branches. The document allows reuse of a projected prior orthogonal direction when opposite corrected movement is at most 25% of its prior injection and its projection norm is at least .5. \[6\]

**Pure orthogonal (rows 13–14).** Use the orthogonal construction without a reversal branch, with dynamic or inverse-log timing as labeled. These rows contain no scores; their separate tested implementation settings are not supplied.

**Response-switching hybrid (rows 15–16; label-to-source match inferred).** Start orthogonal. For an orthogonal attack, compare the next observed movement with its predicted mean injection: 

$$
\gamma_t=\frac{v_t^\top(q_{t+1}-q_t)}{\widetilde w_7\sqrt{R_7^2-m_7^2}+\widetilde w_8\sqrt{R_8^2-m_8^2}}.
$$

 If the median of three completed responses is below $.25$, permanently switch to $a_i=-(R_i/2)\operatorname{unit}(d_7+d_8)$, with orthogonal fallback for a zero sum or no-op. Row 15 is consistent with the supplied inverse-log hybrid; row 16’s “strong stall” wording is not uniquely resolved. The dynamic and inverse-log hybrid documents differ in timing. Weak response does not identify the aggregator. \[7\]

**Max Stall (rows 18, 20–22).** Both *errors* oppose the current honest sum: 

$$
v=-\operatorname{unit}(d_7+d_8),\qquad a_i=d_i+R_iv.
$$

 If the sum is near zero, reuse the last valid direction or the first coordinate. Reduce client 8’s error length by $10^{-6}R_8$ if the submissions coincide. Row 18 uses dynamic timing; rows 20–22 change only timing. \[8,2 pp.46–55\]

**Four-history Stall (row 19, no score).** The available dynamic code uses $v=-\operatorname{unit}(\frac14\sum_{r=t-4}^{t-1}H_r)$ and $a_i=d_i+R_iv$, with four-round warmup. The screenshot labels an inverse-log variant, requested in the conversation; its finalized schedule after warmup is not shown. \[2 pp.38–45,56\]

**Clientwise hybrid (row 24).** Client 7 uses $\Pi_{7,t}(-d_{7,t-1})$; client 8 uses $\sqrt{R_{8,t}^2-m_{8,t}^2}\,v$. Dynamic timing applies to the pair. It does not use the earlier hybrid’s response switch or persistence test. See page 5. \[2 pp.62–69\]

**Adaptive mix (row 25).** Choose among casewise, max-Stall, and orthogonal candidates with dynamic timing. After trying each, select the largest $Q_j+0.4\sqrt{\log(1+\sum_k n_k)/n_j}$. Update $Q_j$ from clipped “lost forward progress + half sideways displacement” feedback (first rate 1, then .3). This feedback is a model-movement proxy, not F1. \[2 pp.70–81\]

**Paper-inspired (row 17).** Only a description of a spike-threshold implementation with inverse-log timing is present. The referenced paper and code are not attached; no mathematical rule can be verified. \[2 pp.28–30\]

## 5. Recorded results

All numbers below are transcribed from the supplied screenshot, preserving displayed precision. A dash means a blank cell, not zero or an inferred failure. Method names are lightly normalized; “added” is retained as a screenshot note.

| Row | Approach                          | Timing / variant                  |                 Score | Note  |
|----:|:----------------------------------|:----------------------------------|----------------------:|:------|
|   2 | Noise                             | Inverse log                       |         $-0.18516232$ |       |
|   3 | Stall                             | Inverse log                       |          $1.32006308$ | added |
|   4 | Echo                              | Inverse log                       |         $-0.18452121$ |       |
|   5 | Casewise                          | Inverse log                       |           $0.1893827$ |       |
|   6 | Stall                             | Custom                            |          $0.75591666$ |       |
|   7 | Echo                              | Custom                            |            $0.001837$ |       |
|   8 | Noise                             | Custom                            |         $-0.18617138$ |       |
|   9 | Shared Stall                      | Inverse log                       |          $0.00427737$ |       |
|  10 | Shared Stall                      | Uniform                           |         $-0.37050763$ |       |
|  11 | Current cancellation              | State-based gates                 |         $-0.18516232$ |       |
|  12 | Orthogonal + Stall                | Dynamic                           |          $0.74969586$ |       |
|  13 | Pure orthogonal                   | Dynamic                           |                     — |       |
|  14 | Pure orthogonal                   | Logarithmic scale                 |                     — |       |
|  15 | Mixed                             | Log                               |          $-0.3717912$ |       |
|  16 | Strong Stall                      | Dynamic                           |                     — |       |
|  17 | Paper-inspired                    | Not stated in table               |         $-0.00018299$ |       |
|  18 | Max Stall                         | Dynamic                           |           $1.1175311$ | added |
|  19 | Stall based on last 4             | Inverse log                       |                     — |       |
|  20 | Max Stall                         | First 30 rounds                   |          $-0.0000015$ |       |
|  21 | Max Stall                         | First 15, then 15 alternating$^*$ |          $0.55942751$ |       |
|  22 | Max Stall                         | Inverse log                       |                     — |       |
|  23 | Initial Stall                     | Dynamic                           |          $1.49751372$ | added |
|  24 | **Clientwise orthogonal + Stall** | **Dynamic**                       | $\mathbf{1.50005821}$ | added |
|  25 | Mix (“mix got”)                   | Dynamic in code                   |          $1.30577048$ | added |

$^*$The screenshot says “max stall first 15 runs and then 15 runs sequentially.” The corresponding conversation/code specifies rounds 1–15, then 17,19,…,45. Timing supplied from code for rows 11 and 25 is identified as such; their screenshot timing cells are blank.

## What these results support

The clientwise hybrid has the highest recorded score, followed by original Stall with dynamic timing. Their difference is only 

$$
1.50005821-1.49751372=\boxed{0.00254449\text{ percentage points}}.
$$

 This establishes the ranking of the supplied aggregate scores, not a statistically reliable advantage. The screenshot provides no repeat-run uncertainty or full per-case breakdown for these two methods. Max-Stall timing also varies substantially in this record: dynamic gives $1.1175311$, compared with approximately zero for the first-30 schedule.

## Evidence boundaries

Exact formulas are taken from the attached formulation PDFs or code printed in the conversation. Rows 12 and 15–16 have abbreviated labels that do not uniquely identify a saved implementation; their source matches are marked above. Missing custom-Noise settings, the paper-inspired formula, and the unscored variants are not filled in with invented details. No scores are attributed to synthetic validation checks.

## 6. Best method: clientwise Stall + orthogonal

**Intuition.** One client opposes its own previous honest step; the other redirects its entire resultant away from the current pair and recent corrected history. Both share one dynamic timing decision. The following pseudocode follows the code printed on pages 62–69 of *Fix Attack Code.pdf*, including its history checks and numerical guard.

|     | **Algorithm: one call at round $t$**                                                                                                                                                                                                                                                  |
|----:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   1 | **Inputs:** $q_t$, honest $d_{7,t},d_{8,t}$, round $t$, persistent state. Initialize $N=0$, $\tau=.025$ (or floored calibration mean), empty history buffers and no previous round. Use weights $(1/8,1/8)$ unless supplied.                                                          |
|   2 | Require finite real inputs and strictly increasing rounds. Let `ready` mean the previous call was at $t-1$. If ready, compute $H_{t-1}=q_t-q_{t-1}-\sum_{i=7}^8\widetilde w_{i,t-1}a_{i,t-1}$ and retain the last three $H$ vectors. On a gap, clear corrected and magnitude history. |
|   3 | Compute $m_i=\left\lVert d_i\right\rVert_2$, $R_i=2\max(m_i,.01)$, $M=(m_7+m_8)/2$; set $a_i=d_i$.                                                                                                                                                                                    |
|   4 | **If** $1\le t<60$, `ready`, $N<30$, and $[M\ge\tau\ \textbf{or}\ 30-N\ge60-t]$, perform steps 5–9. Otherwise leave the pair honest.                                                                                                                                                  |
|   5 | Set numerical guard $\eta=1-64\epsilon_{\rm mach}$. For client 7, set $x=-d_{7,t-1}$ and $L=\left\lVert x-d_7\right\rVert_2$. If $L\le R_7$, set $a_7=x$; otherwise set $a_7=d_7+\eta R_7(x-d_7)/L$.                                                                                  |
|   6 | Build $Z$ from normalized nonzero columns of $d_7,d_8$ and the retained $H$ vectors. Compute its SVD. Let $Q$ contain left singular vectors whose singular values exceed $\max(\operatorname{shape}(Z))\epsilon_{\rm mach}\sigma_{\max}$. If there are no columns, $Q$ is empty.      |
|   7 | Choose the smallest $j$ maximizing $1-\sum_kQ_{jk}^2$. Set $v=e_j-QQ^\top e_j$, reproject $v\leftarrow v-Q(Q^\top v)$, then normalize. Reject a nonfinite or near-zero norm.                                                                                                          |
|   8 | For client 8, set $\widehat R_8=\eta R_8$, $\rho=\widehat R_8\sqrt{\max(0,1-(m_8/\widehat R_8)^2)}$, and $a_8=\rho v$.                                                                                                                                                                |
|   9 | Verify finite outputs and $\left\lVert a_i-d_i\right\rVert_2\le R_i$ for both clients; require $a_8\ne d_8$. Increment $N$ once. Client 7 may be unchanged if its Stall target was already honest.                                                                                    |
|  10 | **After the decision**, for $1\le t<60$, append $M$ to the three-value timing buffer and update $\tau$ by the rule on page 1, including skipped rounds.                                                                                                                               |
|  11 | Save $q_t$, honest $d_t$, applied $a_t$, current weights, and $t$ on every call. **Return** the two 24-entry replacement updates and state. Reset state for each independent evaluation run.                                                                                          |

**AI Disclosure:** AI was used for code generation and writup formatting
