# HAC confidence intervals for autocorrelation function
This repository contains Python code for the simulations in "An Estimating Equation Approach for Robust Confidence Intervals for Autocorrelations of Stationary Time Series" by Hwang and Vogelsang (2023). The paper develops heteroskedasticity and autocorrelation (HAC) robust approaches to construct confidence interval for autocorrelations for the time series with general stationary serial correlation structures. The Monte Carlo simulations by the code investigate null rejection probabilities, power and computation of confidence intervals by the extensive simulations regarding autocorrelation function.

## Usage
The codes are packaged to conduct Monte Carlo simulations for different values of interest. For example, 

```bash
!python empirical_size_graph.py --DGP "AR1-IID" --lag_set "1" --size_set "100,200,500,2000" --phi_set "0.1,0.3,0.5,0.7,0.9,-0.1,-0.3,-0.5,-0.7,-0.9" --replication "1000"
```

This is 

This sentence uses delimiters to show math inline:  $\sqrt{3x-1}+(1+x)^2$

**The Cauchy-Schwarz Inequality**
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$