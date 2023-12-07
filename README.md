# HAC confidence intervals for autocorrelation function
This repository contains Python code for Monte Carlo simulations in "An Estimating Equation Approach for Robust Confidence Intervals for Autocorrelations of Stationary Time Series" by Hwang and Vogelsang (2023). The paper develops heteroskedasticity and autocorrelation (HAC) robust approaches to construct confidence interval for autocorrelations for the time series with general stationary serial correlation structures. The Monte Carlo simulations by the code investigate null rejection probabilities, power and computation of confidence intervals by the extensive simulations regarding autocorrelation function. You can check the simulation results and read the working paper here. [[Paper Link]](https://taeyoonhwang.s3.us-east-2.amazonaws.com/Taeyoon_Hwang_JMP.pdf)

## Usage
The codes are packaged to conduct Monte Carlo simulations for different values of interest. For example,

```bash
python empirical_size_graph.py --DGP "AR1-IID" --lag_set "1" --size_set "100,200,500,2000" --phi_set "0.1,0.3,0.5,0.7,0.9,-0.1,-0.3,-0.5,-0.7,-0.9" --replication "1000"
```

* `empirical_size_graph.py` provides the simulation results that compare null rejections of our approach with that of existing methods. As you can see in the paper, we include existing methods such as Bartlett formula by Bartlett (1946), Taylor $t$-statistic by Taylor (1984) and Dalla, Giraitis and Phillips (2022), generalized Bartlett formula by Francq and Zakoian (2009). Under our estimating equation apporach, we have various versions of test statistics based on different asymptotics. The traditional approach yields $t$-statistics based on kernel HAC variance estimator with normal critical values. Under fixed-$b$ asymptotics, it is based on kernel HAC variance estimator with fixed-$b$ critical values. Under fixed-K asymptotics we use orthonormal series variance estimator suggested by Sun (2013).

* Data Generating Processes (DGPs)

    You can choose DGP for Monte Carlo simulation by `--DGP "AR1-IID"`

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





<!--
This is 

This sentence uses delimiters to show math inline:  $\sqrt{3x-1}+(1+x)^2$

**The Cauchy-Schwarz Inequality**
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$
-->