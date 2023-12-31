# HAC confidence intervals for autocorrelation function
This repository contains Python code for Monte Carlo simulations in "An Estimating Equation Approach for Robust Confidence Intervals for Autocorrelations of Stationary Time Series" by Hwang and Vogelsang (2023). The paper develops heteroskedasticity and autocorrelation (HAC) robust approaches to construct confidence interval for autocorrelations for the time series with general stationary serial correlation structures. The Monte Carlo simulations by the code investigate null rejection probabilities, power and computation of confidence intervals by the extensive simulations regarding autocorrelation function. For comprehensive explanations about the simulation set-up and analysis of the results, please refer to our working paper, available here: [[Paper Link]](https://taeyoonhwang.s3.us-east-2.amazonaws.com/Taeyoon_Hwang_JMP.pdf)

## Usage
The codes are packaged to conduct Monte Carlo simulations for different values of interest by a single command. For example,

```bash
python empirical_size_graph.py --DGP "AR1-IID" --lag_set "1" --size_set "100,200,500,2000" --phi_set "0.1,0.3,0.5,0.7,0.9,-0.1,-0.3,-0.5,-0.7,-0.9" --replication "1000"
```

* `empirical_size_graph.py` provides Monte Carlo simulation results that compare null rejections of our approach with that of existing methods, under a broad set of data generating processes (DGPs) where
$y_{t}$ follows the $ARMA(1,1)$ process $y_{t}=\phi y_{t-1}+\epsilon_{t}+\theta\epsilon_{t-1}$ and $\epsilon_t$ is innovation and can be defined as below. As you can see in the paper, we include existing methods such as Bartlett formula by Bartlett (1946), Taylor $t$-statistic by Taylor (1984) and Dalla, Giraitis and Phillips (2022), generalized Bartlett formula by Francq and Zakoian (2009). Under our estimating equation apporach, we have various versions of test statistics based on different asymptotics. The traditional approach yields $t$-statistics based on kernel HAC variance estimator with normal critical values. Under fixed-b asymptotics, it is based on kernel HAC variance estimator with fixed-b critical values. Under fixed-K asymptotics we use orthonormal series variance estimator suggested by Sun (2013).

* Data Generating Processes (DGPs)

    You can choose DGP for Monte Carlo simulation by, for example, `--DGP "AR1-IID"`.

    DGP 1: IID : $\epsilon_{t}=u_{t}\sim i.i.d.N(0,1)$.

    DGP 2: MDS : $\epsilon_{t}=u_{t}u_{t-1},$ $u_{t}\sim i.i.d.N(0,1)$..

    DGP 3: GARCH : $\epsilon_{t}=h_{t}u_{t}$ and $h_{t}^{2}%
    =0.1+0.09\epsilon_{t-1}^{2}+0.9h_{t-1}^{2}$, $u_{t}\sim i.i.d.N(0,1)$.

    DGP 4: WN-1 : $\epsilon_{t}=u_{t}+u_{t-1}u_{t-2},$ $u_{t}\sim
    i.i.d.N(0,1)$.

    DGP 5: WN-2: $\epsilon_{t}=u_{t}^{2}u_{t-1}$, $u_{t}\sim i.i.d.N(0,1)$.

    DGP 6: WN-NLMA: $\epsilon_{t}=u_{t-2}u_{t-1}(u_{t-2}+u_{t}+1)$,
    $u_{t}\sim i.i.d.N(0,1)$.

    DGP 7: WN-Bilinear: $\epsilon_{t}=u_{t}+0.5u_{t-1}\epsilon_{t-2}$,
    $u_{t}\sim i.i.d.N(0,1).$

    DGP 8: WN-Gamma : $\epsilon_{t}=u_{t}+u_{t-1}u_{t-2},$ $u_{t}
    =\zeta_{t}-E[\zeta_{t}],$ $\zeta_{t}\sim i.i.d.Gamma(0.3,0.4)$.

    DGP 9: WN-Gamma2 : $\epsilon_{t}=u_{t}-u_{t-1}u_{t-2},$ $u_{t}
    =\zeta_{t}-E[\zeta_{t}],$ $\zeta_{t}\sim i.i.d.Gamma(0.3,0.4)$.

    The command for each DGP is provided as follows from DGP 1 to DGP 9. `AR1-IID`, `AR1-MDS`, `AR1-GARCH`, `AR1-WN`, `AR1-non-md1`, `AR1-NLMA`, `AR1-bilinear`, `AR1-WN-gam-v`, `AR1-WN-gam-v-minus`. The usage, for example, is `--DGP "AR1-WN-gam-v"`. You can replace AR1 by MA1 or ARMA11 to use different ARMA specifications, for example, `ARMA11-NLMA`.

 *  `--lag_set` chooses lags of the autocorrelation functions to be considered in the simulation. For example, `--lag_set "1,2,3"` gives the results for the autocorrelation function at lag 1, 2 and 3.

*   `--size_set` chooses numbers of sample size for the simulation. Setting `--size_set "100,200,500,2000"` gives the results for each sample size.

*   `--replication` sets the number of replication for the simulation.

The next part of the simulations is about finite sample power of the test statistics. The paper uses size-adjusted power to account for the size distortions of the tests. The following single command provides size-adjusted power results using our approach as well as existing methods.
```bash
python power_analysis_graph.py --DGP "AR1-MDS" --lag_set "1" --size_set "100,200,300,500" --null_phi "0.5" --delta_set "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" --replication "3000" --size_adj_replication "3000" --two_sided_sig_level "0.025"
```


<!--
This is 

This sentence uses delimiters to show math inline:  $\sqrt{3x-1}+(1+x)^2$

**The Cauchy-Schwarz Inequality**
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$
-->