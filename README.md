# Wind Power Generation Forecasting & Optimization

[![Course](https://img.shields.io/badge/Course-Optimization%20(MYE008)-blue)](https://www.cse.uoi.gr/)
[![Institution](https://img.shields.io/badge/University-University%20of%20Ioannina-navy)](https://www.uoi.gr/)
[![Author](https://img.shields.io/badge/Author-Aliko%20Muska-orange)](#author)
[![Language](https://img.shields.io/badge/Language-Python-green)](#implementation)

A comprehensive benchmark and analytical study on mathematical optimization algorithms applied to short-term wind power forecasting. This project evaluates derivative-based, derivative-free, and evolutionary optimization techniques to calibrate a 5-parameter nonlinear energy model using real-world meteorological sensor data.

---

## 📌 Project Overview

Accurate short-term wind power prediction (24-hour horizon) is critical for modern power grid management and renewable energy integration into the European power grid (EU Renewable Energy Directive 2023/2413). 

This project optimizes a 5-parameter parametric model connecting atmospheric readings to power output ($E$) measured at 10-minute intervals across 4 training days (576 samples) and evaluates generalization on a 5th unseen test day (144 samples).

### Mathematical Model

The power output function $E(\mathbf{\beta}, v, \theta, T, P)$ is defined as:

$$E(\mathbf{\beta}, v, \theta, T, P) = \beta_0 v^2 + \beta_1 \sin(\theta) + \beta_2 e^{\beta_3 T} + \beta_4 \ln(P)$$

Where:
- $v$: Wind Speed ($	\text{m/sec}$)
- $\theta$: Wind Direction ($0^\circ - 360^\circ$)
- $T$: Temperature ($^\circ	ext{C}$)
- $P$: Atmospheric Pressure ($	ext{hPa}$)
- $\mathbf{\beta} = [\beta_0, \beta_1, \beta_2, \beta_3, \beta_4] \in \mathcal{B}$: Model parameter vector.

### Search Space Bounds $(\mathcal{B})$

$$\mathcal{B} = [10, 100] \times [-10, 10] \times [50, 200] \times [0.01, 0.1] \times [0.1, 1.0]$$

---

## 🧪 Optimization Objective

The objective is to minimize the **Mean Squared Error (MSE)** over the training dataset:

$$\min_{\mathbf{\beta} \in \mathcal{B}} 	ext{MSE}_{	ext{train}}(\mathbf{\beta}) = \frac{1}{576} \sum_{t=1}^{576} \left( E(\mathbf{\beta}, v_t, \theta_t, T_t, P_t) - E_t \right)^2$$

### Key Algorithmic & Implementation Highlights
- **Exact Analytical Derivatives:** Derived complete 1st order gradients ($\nabla 	ext{MSE}$) and 2nd order Hessian matrices ($\nabla^2 	ext{MSE}$) for exact Newton and BFGS computations.
- **Penalty Functions:** Quadratic penalty terms $P(\mathbf{\beta})$ integrated to handle boundary constraints effectively during line searches and trust region updates.
- **Normalized Domain Search ($\tilde{\mathcal{B}} = [0, 1]^5$):** Scaled parameter dimensions to prevent gradient ill-conditioning due to scale disparities across variables.
- **Benchmark Protocol:** 30 independent runs per algorithm starting from standardized initial points, capped at $N_{	ext{max}} = 100,000$ function evaluations per run.

---

## 🛠️ Algorithms Evaluated

1. **NewtonTR**: Newton Trust-Region method with Dogleg step strategy utilizing exact analytical Hessians.
2. **BFGSWolfe**: Quasi-Newton BFGS optimization paired with Wolfe (Armijo sufficient decrease) line search.
3. **NelderMead**: Derivative-free Direct Search Simplex algorithm (Reflection, Expansion, Contraction, Shrinkage).
4. **GA (Genetic Algorithm)**: Binary-encoded Genetic Algorithm utilizing roulette wheel selection, $k$-point/uniform crossover, and point mutation.
5. **PSO (Particle Swarm Optimization)**: Swarm-based heuristic with dynamic neighborhood topology.

---

## 📊 Experimental Results & Statistics

### Performance Summary (Training & Convergence)

All metrics are calculated over 30 independent runs ($N_{	ext{max}} = 100,000$ budget):

| Algorithm | Mean $f_{	ext{best}}$ (MSE) | Median $f_{	ext{best}}$ | Min $f_{	ext{best}}$ | Mean Budget Spent (`last-hit`) | Convergence Speed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BFGSWolfe** | **1207.91** | **108.40** | **108.40** | **2,467 evaluations** | **⚡ Ultra Fast** |
| **NelderMead** | 110.14 | 108.40 | 108.40 | 100,005 evaluations | 🐢 Slow (Exhausted Budget) |
| **GA** | 111.08 | 110.59 | 108.79 | 100,026 evaluations | 🐢 Slow (Exhausted Budget) |
| **NewtonTR** | 129.32 | 7123.98* | 108.42 | 99,996 evaluations | ⚠️ Local minima sensitive |

*Note: BFGS with Wolfe line search demonstrated outstanding speed, converging to optimal solutions (~108.40 MSE) in under 2,500 function evaluations on average.*

---

### Generalization Performance (Test Set Evaluation)

Optimal model parameters were evaluated on unseen Day 5 test data ($144$ time steps):

| Algorithm Model | Test MSE ($	ext{MSE}_{	ext{test}}$) | Performance vs Train |
| :--- | :---: | :--- |
| **GA** | **73.09** | **Best Generalization** |
| **NewtonTR** | 74.38 | Excellent Generalization |
| **BFGSWolfe** | 74.71 | Robust Generalization |
| **NelderMead** | 74.73 | Robust Generalization |

> **Key Finding:** While gradient/quasi-Newton methods (BFGS) achieved rapid training convergence, population-based evolutionary approaches (GA) provided slightly superior generalization performance on unseen test data ($	ext{MSE}_{	ext{test}} = 73.09$).

---

### Statistical Significance (Wilcoxon Rank-Sum Test)

A pairwise Wilcoxon rank-sum test at significance level $lpha = 0.05$ was conducted across the 30 runs:

| Pairwise Comparison | $p$-value | Result | Significance |
| :--- | :---: | :---: | :--- |
| **NewtonTR vs BFGSWolfe** | $0.6015$ | $pprox$ | No statistically significant difference |
| **NewtonTR vs NelderMead** | $0.2506$ | $pprox$ | No statistically significant difference |
| **BFGSWolfe vs NelderMead** | $0.1745$ | $pprox$ | No statistically significant difference |
| **GA vs NelderMead / BFGS** | $0.3472$ | $pprox$ | Statistically comparable solution quality |

---

## 📁 Repository Structure

```text
.
├── data/
│   ├── data_train.txt       # 576 samples (Days 1-4)
│   ├── data_test.txt        # 144 samples (Day 5)
│   └── initial_points.txt   # 30 standardized initial parameter vectors
├── src/
│   ├── functions.py         # Objective function, analytical gradients & Hessians
│   ├── newton_tr.py         # Newton Trust Region with Dogleg solver
│   ├── bfgs.py              # Quasi-Newton BFGS implementation
│   ├── nelder_mead.py       # Nelder-Mead Simplex solver
│   ├── ga.py                # Genetic Algorithm implementation
│   └── pso.py               # Particle Swarm Optimization implementation
├── outputs/
│   ├── output_*_train.txt   # Optimization logs per algorithm
│   └── output_test.txt      # Day 5 generalization test results
├── report.pdf               # Detailed academic report (Greek)
└── README.md                # Project documentation
```

---

## 👤 Author

**Aliko Muska**  
Student AM: 4427  
Department of Computer Science and Engineering  
University of Ioannina  

---
*Developed for the Optimization (MYE008) course, Academic Year 2024–2025.*
