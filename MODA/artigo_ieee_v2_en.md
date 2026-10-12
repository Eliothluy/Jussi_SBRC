<!--
IEEE-oriented manuscript (English), v2-campaign methodology.
Placeholder convention: [PENDING-v2: <what>] marks every number/claim that
must be filled from the v2 campaign (results_controlled/heuristics_metaheuristics_v2/).
Fill-in pipeline: examples/analyze_v2_stats.py + examples/generate_v2_audit_figures.py.
Do NOT mix v1 numbers into this document (AGENTS.md §14.7).
-->

# Constraint-Aware Metaheuristic Optimization of Slice Allocation Weights in 5G/O-RAN Networks

## Abstract

Network slicing requires radio access networks to partition resources among
service classes with conflicting requirements: throughput for eMBB, latency
and reliability for URLLC, and massive connectivity for MTC. We formulate
inter-slice resource allocation as a continuous optimization problem over a
weight vector on the probability simplex, in which each weight controls one
slice's share of the physical resource block (PRB) budget, while conventional
schedulers (proportional fair) remain in charge of intra-slice allocation. We
compare four metaheuristics — Particle Swarm Optimization (PSO), a Genetic
Algorithm (GA), Simulated Annealing (SA), and a hybrid method combining
population-based exploration with stochastic local refinement — under a
**feasibility-first objective** that treats slice service-level agreements
(SLAs) as hard constraints: URLLC packet-level 99th-percentile delay,
per-slice packet delivery ratio, and eMBB offered-load satisfaction. Unlike
weighted-sum formulations, this objective cannot reward configurations that
starve a slice. Each candidate is evaluated by a full-stack 5G simulation
(ns-3 / 5G-LENA) with per-packet latency instrumentation, across
[PENDING-v2: 4] traffic scenarios and 10 independent seeds, under an
equalized evaluation budget of 48 evaluations per method. Paired
non-parametric tests (Friedman, Wilcoxon signed-rank) over per-seed best
scores show that [PENDING-v2: main statistical finding]. Under congestion,
the best evolved weight vectors [PENDING-v2: headline physical result —
deadline-violation rate / worst-slice SLA vs. baselines]. All code,
scenarios, and per-seed artifacts are available for reproduction.

**Keywords:** network slicing; 5G; O-RAN; metaheuristics; constrained
optimization; particle swarm optimization; genetic algorithms; simulated
annealing; resource allocation.

## I. Introduction

Fifth-generation (5G) networks support network slicing natively, allowing
heterogeneous service classes to share one physical infrastructure under
distinct service-level agreements (SLAs). Enhanced Mobile Broadband (eMBB)
slices prioritize sustained throughput; Ultra-Reliable Low-Latency
Communication (URLLC) slices prioritize bounded latency and delivery
reliability; massive Machine-Type Communication (MTC) slices prioritize
serving many low-rate devices without starvation. Because these objectives
conflict whenever radio resources are scarce, inter-slice allocation is a
multi-objective decision problem whose difficulty grows with load.

O-RAN architectures encourage programmable control loops (rApps/xApps) that
adjust radio-layer policy without replacing the underlying schedulers. A
pragmatic design keeps a standard intra-slice scheduler (e.g., proportional
fair) and optimizes, at the inter-slice level, a small continuous vector of
allocation weights that partitions the PRB budget among slices. This
separation retains compatibility with production schedulers and reduces the
search space to a low-dimensional simplex — but the mapping from weights to
network KPIs remains non-convex, noisy, and only accessible through
simulation or measurement, which motivates black-box metaheuristic search.

A central methodological problem, often overlooked, is the **objective
function**. Weighted-sum scores over KPI aggregates can assign high scores to
degenerate allocations: in a preliminary version of this study, a
weighted-sum objective ranked *first* a configuration that effectively
starved the MTC slice (packet delivery ratio of 2.4%, mean delay above 4 s)
because a throughput-only SLA indicator saturated at 100%. This is not an
implementation accident but a structural failure mode of soft-penalty
scalarizations. We therefore adopt a feasibility-first formulation in which
SLA violations are hard constraints and infeasible candidates are always
dominated by feasible ones, and we instrument the simulator with
**per-packet** latency percentiles so that tail-latency constraints are
measured rather than approximated by window averages.

The contributions of this paper are:

1. A constraint-aware formulation of inter-slice weight optimization on the
   simplex, with hard SLA constraints on URLLC tail latency (per-packet p99),
   per-slice delivery ratio, and eMBB offered-load satisfaction.
2. A controlled comparison of PSO, GA, SA, and a hybrid metaheuristic under
   an equalized evaluation budget (48 evaluations per method) across
   [PENDING-v2: 4] traffic scenarios and 10 independent seeds, with paired
   non-parametric statistical testing.
3. A per-packet measurement methodology for slice SLAs in ns-3/5G-LENA —
   packet-level delay percentiles (p95/p99/p99.9), deadline-violation rate,
   and in-time reliability — replacing window-sampled statistics that
   systematically underestimate tails.
4. A physical-KPI evaluation of the best evolved allocations against pure
   schedulers (RR, PF, BCQI), slice-aware baselines, and adaptive heuristics
   (including AQPS), showing when weight optimization pays off and when it
   does not.

## II. Related Work

[PENDING-v2: related-work section — RSLAQ (DRL-based SLA control in O-RAN),
AQPS (QoS-aware priority scheduling), metaheuristics for RAN resource
allocation, network slicing surveys. The workspace PDFs provide RSLAQ and
AQPS metadata; complete after reference consolidation.]

Distinctly from DRL approaches such as RSLAQ, which learn a control policy
online, our approach searches offline for a static per-scenario weight
configuration; it requires no training infrastructure, is auditable (every
evaluated candidate is persisted with its full KPI record), and its outputs
can seed either static configuration (Non-RT RIC time scale) or serve as
ground truth for learned policies. It is *not* a Near-RT control mechanism:
each candidate evaluation costs a full simulation, which confines
metaheuristic search to offline/periodic optimization.

## III. Problem Formulation

### A. Decision variables

Let $S = 3$ slices indexed by $i \in \{$eMBB, URLLC, MTC$\}$. A candidate
solution is the continuous weight vector

$$\mathbf{w} = [w_{\mathrm{eMBB}}, w_{\mathrm{URLLC}}, w_{\mathrm{MTC}}], \qquad w_i \ge 0, \qquad \sum_i w_i = 1,$$

i.e., a point on the 2-simplex. The MAC scheduler partitions the per-slot
resource block group (RBG) budget proportionally to $\mathbf{w}$ among slices
with active demand (budget unused by inactive slices is redistributed), and a
proportional-fair policy allocates within each slice. Candidates produced by
the optimizers are projected back onto the simplex by clipping negative
components and renormalizing.

### B. Constraint-aware objective

Each candidate is evaluated by one end-to-end simulation, from which
per-slice KPIs are extracted. The objective is *feasibility-first*
maximization:

$$
F(\mathbf{w}) =
\begin{cases}
\dfrac{100}{S}\sum_{i} \mathrm{SLA}_i(\mathbf{w}) & \text{if } V(\mathbf{w}) = 0,\\[2mm]
-100\, V(\mathbf{w}) & \text{otherwise,}
\end{cases}
$$

where $V(\mathbf{w}) \ge 0$ is the sum of normalized violations of the hard
constraints

$$
D^{99}_{\mathrm{URLLC}} \le 10\,\mathrm{ms}, \qquad
\mathrm{PDR}_{\mathrm{URLLC}} \ge P^{\min}_{U}, \qquad
\mathrm{PDR}_{\mathrm{MTC}} \ge P^{\min}_{M}, \qquad
T_{\mathrm{eMBB}} \ge \phi \cdot L_{\mathrm{eMBB}},
$$

with $D^{99}$ the **per-packet** 99th-percentile downlink delay, PDR the
end-to-end packet delivery ratio, $T$ the delivered throughput, $L$ the
offered load, and $\phi$ the minimum served fraction. $\mathrm{SLA}_i \in
[0,1]$ is a composite per-slice satisfaction, $\mathrm{SLA}_i = \min(S^{thr}_i,
S^{pdr}_i, S^{delay}_i)$ — the *worst* dimension governs, so no slice
dimension can compensate another. Two properties follow by construction:
(i) any feasible candidate dominates every infeasible one; (ii) among
infeasible candidates the search still receives a gradient toward
feasibility, avoiding flat penalty regions.

Constraint thresholds were calibrated on a pilot campaign so that every
scenario retains a non-empty feasible region:
[PENDING-v2: final calibrated values of $P^{\min}_U$, $P^{\min}_M$, $\phi$ —
current pre-calibration defaults: 90%, 80%, 0.60. Report the calibration
procedure outcome here.]

This formulation operationalizes the URLLC latency budget as a measured
packet-level constraint rather than a soft penalty on mean delay — a design
decision motivated by the failure mode described in Section I.

## IV. Optimization Methods

All methods maximize the same $F(\mathbf{w})$, receive the same evaluation
budget, and share the checkpointed evaluation pipeline (Section V-D).

**PSO.** A swarm of 6 particles with inertia 0.55 and cognitive/social
coefficients 1.30; positions are renormalized onto the simplex after each
update. 8 iterations × 6 particles = 48 evaluations.

**GA.** Population 6 with elitism, blend crossover on the simplex, and
Gaussian mutation (σ = 0.12) followed by renormalization. 8 generations × 6
individuals = 48 evaluations.

**SA.** Single-trajectory annealing with Gaussian perturbations and a
decreasing temperature schedule. To equalize budgets, the chain length is set
to 48 evaluations (matching GA/PSO), not to the iteration count.

**Hybrid.** A population method that combines an elite archive, PSO-inspired
updates, GA-style recombination, and one SA-style local refinement of the
global best per iteration. Because of the extra local step it consumes 56
evaluations (8 × (6+1)); comparisons at the common 48-evaluation ceiling are
also reported to keep the budget comparison fair.
[PENDING-v2: verify both budget readings appear in the results tables.]

The optimizer RNG is seeded independently per simulation seed
(`search_seed = base_seed + ns3_seed`), so the 10 repetitions are independent
searches, not one trajectory replayed on 10 channels.

## V. Experimental Methodology

### A. Simulation environment

Experiments use ns-3 with the 5G-LENA NR module. A single gNB (3.55 GHz,
100 MHz, numerology μ=1, TDD `D|D|8D|4GB|4U|U|U`, RLC UM, 43 dBm) serves
stationary UEs placed on a disk of radius 100 m; each UE receives a downlink
UDP flow from a remote host through the core network. UEs are grouped into
the three slices in contiguous ID ranges; FlowMonitor maps flows back to
slices by destination port. A custom slice-aware MAC scheduler partitions the
RBG budget according to $\mathbf{w}$ (Section III-A).

### B. Traffic scenarios

[PENDING-v2: confirm final scenario table — 4 scenarios in the main paper
(low_traffic, normal, congestion, stressed); a fifth profile
(insufficient_resources) is simulated and reported as a robustness check.]

| Scenario | UEs (eMBB/URLLC/MTC) | Offered load (Mb/s) |
|---|---|---|
| low_traffic | 2 / 2 / 6 | 57 |
| normal | 5 / 5 / 10 | 73 |
| stressed | [PENDING-v2] | 128 |
| congestion | 15 / 10 / 35 | 245 |

These are nominal input profiles; the operational regime (whether the cell is
actually congested) is diagnosed from measured PDR and delay, not assumed
from the label. Simulation time is 15 s per evaluation with 0.4 s application
warm-up and 0.2 s drain (KPIs are computed over the active window only).

### C. Per-packet SLA instrumentation

Latency statistics are computed from the FlowMonitor **per-packet delay
histogram** (bin width 0.5 ms), merged per slice: percentiles p95/p99/p99.9
are interpolated within bins; the *deadline-violation rate* is the fraction
of delivered packets exceeding the slice's latency budget; *in-time
reliability* is the fraction of transmitted packets delivered within budget
(undelivered packets count as violations). Packet loss is reported end to end
($\mathrm{PLR} = (tx - rx)/tx$, so $\mathrm{PDR} + \mathrm{PLR} = 100\%$ by
construction), with detected-loss (FlowMonitor) kept as a separate diagnostic
column. We emphasize this because window-averaged sampling — used in a
preliminary version of this pipeline — produced `mean > p95` artifacts in
about one fifth of the measured cells and systematically hid tail latency;
all results in this paper use per-packet statistics.

### D. Campaign protocol and reproducibility

For every (scenario, seed) pair, all four methods run under the same
objective, the same seeds (1–10), and the same simulator build. Every
candidate evaluation is persisted with a JSON sidecar (weights, score,
objective version, full KPI record), which makes the search resumable and the
comparison auditable a posteriori. Baselines (pure RR/PF/BCQI, slice-aware
variants, adaptive heuristics including AQPS) run under the identical
protocol and seeds. [PENDING-v2: total evaluation counts and total compute
time of the campaign.]

### E. Statistical analysis

The experimental unit is the **best score per (method, scenario, seed)** —
one independent search outcome per seed, 10 per method/scenario. We report
mean ± sd, median, and bootstrap 95% CIs; the omnibus Friedman test across
the four methods; and pairwise Wilcoxon signed-rank tests with rank-biserial
effect sizes. With $n = 10$ paired seeds these tests are adequately powered
for large effects; we do not claim significance where $p \ge 0.05$.

## VI. Results

> All numbers below come exclusively from the v2 campaign
> (`results_controlled/heuristics_metaheuristics_v2/`). Fill via
> `analyze_v2_stats.py` (tables) and `generate_v2_audit_figures.py` (figures).

### A. Feasibility structure of the search space

[PENDING-v2: per scenario — fraction of evaluated candidates that satisfy all
hard constraints; which constraint binds most often (expected: URLLC p99
under congestion); whether any scenario has an empty feasible region after
calibration. This subsection is new relative to weighted-sum studies and is
the direct payoff of the feasibility-first design.]

### B. Method comparison under equal budget

[PENDING-v2: Table — best score per seed: mean ± sd, median, CI95 per
method × scenario (from v2_best_per_seed.csv); Friedman χ²/p per scenario;
pairwise Wilcoxon p + effect sizes. Figure: per-seed dispersion strips
(fig_v2_6) and score bars (fig_v2_1).]

[PENDING-v2: Convergence — evaluations to reach 95% of final best, per
method (the pilot's convergence justifies the 48-evaluation budget; report
the v2 numbers).]

### C. Physical SLA outcomes of the best allocations

[PENDING-v2: Table/figures — URLLC p99 and p99.9 (fig_v2_2), deadline
violation rate (fig_v2_3), worst-slice SLA (fig_v2_4), MTC PDR, per method.
Key question the audit posed: does ANY method satisfy all SLAs
simultaneously, and at what load does that stop being possible?]

### D. Comparison with schedulers and heuristics

[PENDING-v2: best metaheuristic vs pure RR/PF/BCQI, slice-aware baselines,
AQPS and adaptive heuristics — same protocol, same seeds. Physical Pareto
view: total throughput × URLLC p99 (fig_v2_5). The v1 pilot suggested pure
BCQI reaches the best URLLC tail at the cost of MTC starvation; verify
whether feasibility-constrained search finds allocations that dominate it.]

### E. Weight stability across seeds

[PENDING-v2: violin plot of best-candidate weights per slice across seeds
(fig_v2_7); coefficient of variation per scenario. This measures whether the
optimum is a property of the scenario or of the noise realization — a
robustness question weighted-sum studies typically skip.]

## VII. Discussion

[PENDING-v2: rewrite around the actual findings. Anchor points to address:]

1. *When does weight optimization pay off?* The preliminary evidence
   indicates gains concentrate under resource contention; quantify with v2.
2. *Feasibility vs. score.* The feasibility-first objective changes the
   qualitative behavior of the search — document how the recommended
   allocations differ from the weighted-sum optima (which starved MTC).
3. *Practical deployment.* Search cost per scenario ([PENDING-v2] CPU-hours)
   confines the approach to Non-RT RIC time scales; discuss the xApp path
   (offline-optimized weights served as policy) and its relation to learned
   policies.
4. *Method choice.* [PENDING-v2: whether any method is statistically
   distinguishable at n=10; if not, say so plainly — a negative result with
   adequate testing is publishable and honest.]

## VIII. Threats to Validity

1. **Simulation scope.** Single cell, stationary UEs, downlink UDP only;
   topology and load co-vary across scenarios. Results characterize
   inter-slice weight optimization in this controlled setting, not deployed
   networks.
2. **Statistical power.** $n = 10$ paired seeds supports detection of large
   effects only; small method differences may remain undetected. We report
   effect sizes and CIs alongside p-values.
3. **Budget asymmetry.** The hybrid consumes 56 evaluations vs. 48 for the
   others; both the full-budget and common-ceiling comparisons are reported.
4. **Threshold calibration.** Hard-constraint thresholds were calibrated on a
   pilot (congestion) and held fixed; different SLA contracts would change
   the feasible region and possibly the ranking. The calibration procedure
   and sensitivity are documented.
5. **Evaluation noise.** A single simulation evaluates each candidate; seed
   effects are handled across, not within, searches. [PENDING-v2: if any
   evaluation timeouts/crashes occurred, report the count and the
   hard-penalty handling.]
6. **Measurement caveats.** FlowMonitor throughput includes IP/UDP headers;
   delay histograms have 0.5 ms resolution (percentile error bounded by bin
   width); histogram-based percentiles are interpolated.

## IX. Conclusion

[PENDING-v2: write after results. Required elements: (i) the
constraint-aware formulation and per-packet instrumentation as the
methodological contribution; (ii) the statistically-tested method comparison
(state clearly if methods are indistinguishable); (iii) the physical-KPI
verdict — does optimized slicing satisfy all SLAs simultaneously, and under
which loads; (iv) future work: online adaptation (xApp), multi-cell,
mobility, dataset-driven learned policies distilled from the search
(offline-RL line).]

## References

[PENDING-v2: consolidate BibTeX. Minimum set: RSLAQ paper; AQPS paper; O-RAN
architecture reference; 5G-LENA/ns-3 simulator papers (Koutlia et al.);
canonical PSO (Kennedy & Eberhart 1995), GA (Holland/Goldberg), SA
(Kirkpatrick 1983); a network-slicing survey; a metaheuristics-in-wireless
survey; statistical methodology (Demšar 2006 for classifier comparison
protocol with Friedman/Wilcoxon).]
