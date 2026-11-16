# Event-stream benchmark metrics

$P$ = prediction, $G$ = recorded ground truth (GT), $E$ = either stream. Both streams use the same crop and scoring interval. Descriptors are computed on one stream and can be compared with the GT descriptor. For paired errors, zero is best.

**Abbreviations**

| Term | Full name or meaning |
|---|---|
| ISI / IEI | Inter-spike interval / inter-event interval. Here both use consecutive events at the same pixel. |
| CV / CDF | Coefficient of variation / cumulative distribution function. |
| MAE / RMSE | Mean absolute error / root mean squared error. |
| MSE / PMSE | Mean squared error / pooled mean squared error. |
| Relative L1 | Relative L1 error (sum of absolute errors divided by the reference L1 norm). |
| Same-ts | Same timestamp. |
| ON / OFF | Positive / negative event polarity. |

## 1. Count, rate and polarity

| Metric | Formula | Meaning |
|---|---|---|
| `event_count` | $N_E=\sum_{e\in E}1$ | Total events in the interval. |
| `event_rate` | $r_k=n_k/\Delta t_k$, $r_E=N_E/T$ | Events/s. The saved array contains bin rates and the CSV gives the whole-sequence rate. |
| `on_percentage` | $100N_+/\max(N_E,1)$ | Percentage of ON events. |
| `event_count_ratio` | $N_P/\max(N_G,1)$ | Prediction-to-GT count ratio. With nonempty GT, 1 is ideal. |

**Symbols.** $e$ = event, $N_E$ = stream count, $N_+$ = ON count, $n_k$ = count in bin $k$, $\Delta t_k$ = bin duration in seconds, $T$ = sequence duration in seconds.

## 2. Event timing

| Metric | Formula | Meaning |
|---|---|---|
| `isi_distribution` | $\delta_{u,j}=1000(t_{u,j+1}-t_{u,j})$ | Array of consecutive same-pixel intervals in ms, pooled across pixels and polarities. Includes zero intervals. |
| `isi_median_ms` | $\mathrm{median}(D)$ | Median of the pooled intervals in ms. |
| `isi_cv` | $\sigma(D)/\max(\mu(D),10^{-9})$ | Relative spread of the pooled intervals. Median and CV return 0 for an empty interval array. |
| `iei_distance` | $\int_{\mathbb R}\lvert F_P(z)-F_G(z)\rvert\,dz$ | Wasserstein-1 distance between log-interval distributions. Undefined if either stream has no positive interval. |
| `same_ts` | $\sum_{w,q:\,c_{wq}>1}c_{wq}/N_E$ | Fraction of events sharing a timestamp with another event anywhere in the same window. Undefined for an empty stream. |

**Symbols.** $t_{u,j}$ = time in seconds of event $j$ at pixel $u$, $D$ = pooled array of $\delta_{u,j}$, $\mu,\sigma$ = population mean and standard deviation. $F_P,F_G$ = empirical CDFs of $z=\log_{10}(\max(\delta_{u,j}/1000,10^{-6}))$ using positive intervals only. $c_{wq}$ = count at rounded 1 ns tick $q$ in 50 ms window $w$.

## 3. Local count variability

**`fano_factor`** compares fluctuations in event counts within each active tile. The CSV reports the mean absolute log ratio over tile-windows active in both streams.

$$
F_i(E)=\frac{\sigma^2_j(c_{ij})}{\mu_j(c_{ij})},
\qquad
\mathrm{error}=\frac{1}{\lvert S\rvert}\sum_{i\in S}
\left\lvert\ln\frac{F_i(P)+10^{-3}}{F_i(G)+10^{-3}}\right\rvert .
$$

**Symbols.** $i$ = 8×8 pixel tile in a 50 ms window, $c_{ij}$ = count in its $j$-th 2 ms bin, $\mu_j,\sigma^2_j$ = mean and population variance over time bins, $S$ = tile-windows with events in both streams. Different active masks give `partial_support` when the score is finite. Empty overlap gives an undefined score.

## 4. Spatiotemporal event distance

**`chamfer_full`** compares every event with its nearest same-polarity event in the opposite stream. The score averages both directions, then gives equal weight to each shared polarity (pixel-equivalent units).

$$
d(A,B)=\frac{1}{\lvert A\rvert}\sum_{a\in A}\min_{b\in B}\lVert a-b\rVert_2,
\quad C_s=\tfrac12[d(P_s,G_s)+d(G_s,P_s)],
\quad C=\frac{1}{\lvert S\rvert}\sum_{s\in S}C_s .
$$

**Symbols.** $a,b=(x,y,v\,t_{\rm ms})$ denote event coordinates, with $x,y$ in pixels, $t_{\rm ms}$ in milliseconds and $v=1$ pixel/ms by default. $P_s,G_s$ = event collections of polarity $s$ (ON or OFF), $A,B$ = the two collections in query/target order, $S$ = polarities present in both. `chamfer_on` and `chamfer_off` store $C_s$. A missing component is undefined, as is $C$ when $S$ is empty.

## 5. Voxel-grid errors

All voxel errors compare aligned prediction and GT grids. Count metrics use signed counts (ON minus OFF) for the main CSV value and include inactive cells.

| Metric | Formula | Meaning |
|---|---|---|
| `voxel_mae` | $\frac1M\sum_i\lvert e_i\rvert$ | Mean absolute count error per cell. |
| `voxel_rmse` | $\sqrt{\frac1M\sum_i e_i^2}$ | Root mean squared count error per cell. |
| `voxel_relative_l1` | $\sum_i\lvert e_i\rvert/\sum_i\lvert G_i\rvert$ | Absolute error relative to GT mass on the signed grid. Undefined at zero GT mass. |
| PMSE, $k=2,4$ | $\frac{\sum_{s,b}w_b(\bar P_{s,b}-\bar G_{s,b})^2}{2\sum_b w_b}$ | MSE of block means on forward-interpolated ON and OFF grids, pooled separately. |

**Symbols.** $i$ = voxel index, $M$ = number of evaluated cells, $e_i=P_i-G_i$ = count error. $b$ = nonoverlapping block of up to $k\times k\times k$ cells (time, height, width), $s$ = polarity, $\bar P_{s,b},\bar G_{s,b}$ = block means, $w_b$ = block weight.

| Protocol | Time bins | PMSE boundary rule |
|---|---|---|
| `run_metric.py` | By default, 160 equal bins over the scoring interval. Legacy frame-pair bins are optional for count errors. | `pmse_2_v2ce`, `pmse_4_v2ce` clip interpolation at the endpoint and average complete blocks equally ($w_b=1$). |
| `run_voxel_metric_v2.py` | Fixed width in ms, with a possibly shorter final bin. | `pmse_2_v2ce_adapted`, `pmse_4_v2ce_adapted` retain last-bin mass and partial blocks ($w_b$ = actual cell count). |

V2 also reports MAE/RMSE on ON, OFF and total (ON + OFF) counts. Its polarity-aware MAE is the mean of the two channel MAEs. Its polarity-aware RMSE is the root mean square of the two channel RMSEs.

[Event metrics](metrics.py) · [Main runner](run_metric.py) · [Voxel metrics](voxel_metric.py) · [Voxel V2](voxel_metric_v2.py)
