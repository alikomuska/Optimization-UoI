# Wind Power Generation Forecasting & Optimization

[![Course](https://img.shields.io/badge/Course-Optimization%20(MYE008)-blue)](https://www.cse.uoi.gr/)
[![Institution](https://img.shields.io/badge/University-University%20of%20Ioannina-navy)](https://www.uoi.gr/)
[![Author](https://img.shields.io/badge/Author-Aliko%20Muska-orange)](#author)
[![Language](https://img.shields.io/badge/Language-Python-green)](#implementation)

A comprehensive benchmark and analytical study on mathematical optimization algorithms applied to short-term wind power forecasting. This project evaluates derivative-based, derivative-free, and evolutionary optimization techniques to calibrate a 5-parameter nonlinear energy model using real-world meteorological sensor data.

---

## 📌 Project Overview

Accurate short-term wind power prediction is critical for modern power grid management and renewable energy integration into the European power grid (EU Renewable Energy Directive 2023/2413). 

This project optimizes a 5-parameter parametric model connecting atmospheric readings to power output ($E$) measured at 10-minute intervals across 4 training days (576 samples) and evaluates generalization on a 5th unseen test day (144 samples).

### Mathematical Model

The power output function $E(\mathbf{\beta}, v, \theta, T, P)$ is defined as:

$$E(\mathbf{\beta}, v, \theta, T, P) = \beta_0 v^2 + \beta_1 \sin(\theta) + \beta_2 e^{\beta_3 T} + \beta_4 \ln(P)$$

Where:
- $v$: Wind Speed (${m/sec}$)
- $\theta$: Wind Direction ($0^\circ - 360^\circ$)
- $T$: Temperature ($^\circ	{C}$)
- $P$: Atmospheric Pressure ($hPa$)
- $\mathbf{\beta} = [\beta_0, \beta_1, \beta_2, \beta_3, \beta_4] $: Model parameter vector.

### Search Space Bounds $(\mathcal{B})$

$$\mathcal{B} = [10, 100] \times [-10, 10] \times [50, 200] \times [0.01, 0.1] \times [0.1, 1.0]$$

---

## 🧪 Optimization Objective

The objective is to minimize the **Mean Squared Error (MSE)** over the training dataset:

$$\min_{\mathbf{\beta} \in \mathcal{B}} 	ext{MSE}_{	ext{train}}(\mathbf{\beta}) = \frac{1}{576} \sum_{t=1}^{576} \left( E(\mathbf{\beta}, v_t, \theta_t, T_t, P_t) - E_t \right)^2$$

### Key Algorithmic & Implementation Highlights
- **Exact Analytical Derivatives:** Derived complete 1st order gradients ($\nabla 	ext{MSE}$) and 2nd order Hessian matrices ($\nabla^2 	ext{MSE}$) for exact Newton and BFGS computations.
- **Penalty Functions:** Quadratic penalty terms $P(\mathbf{\beta})$ integrated to handle boundary constraints effectively during line searches and trust region updates.
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



*Developed for the Optimization (MYE008) course, Academic Year 2024–2025.*
