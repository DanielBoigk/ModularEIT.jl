---
tags: [machine-learning, architecture]
aliases: [Neural ODE, Continuous normalizing flows]
---

A **neural ordinary differential equation** defines a network as the flow of a learned vector field:

$$
\frac{\mathrm dh}{\mathrm dt} = f_\theta(h(t),t),\qquad h(0) = x,\qquad \text{output } h(T).
$$

It is the continuous-depth limit of a residual network $h_{k+1} = h_k+\Delta t\,f_\theta(h_k,t_k)$.

**Training.** Gradients of a loss $\ell(h(T))$ with respect to $\theta$ come from the **adjoint sensitivity method**: solve the adjoint ODE $\dot a = -a^\top\partial_hf_\theta$ backwards from $a(T) = \partial\ell/\partial h(T)$, and accumulate $\int_0^T a^\top\partial_\theta f_\theta\,\mathrm dt$. This is the same idea as the PDE [[Adjoint State Method]]. Memory is constant in depth, but the backward solve adds cost and can suffer from numerical error unless checkpointing or discrete adjoints are used.

**As image restorers.** A neural ODE can map a degraded image to a clean one by integrating a learned "restoration flow". Randomising the integration time $T$ during training makes the flow more robust to the unknown degradation strength. A drawback is cost: every forward pass is an ODE solve with many evaluations of a CNN, and training differentiates through it. Deterministic regression to the mean also tends to produce **blurry** outputs, because it predicts the conditional mean. Generative models such as [[Diffusion Models]] avoid this by sampling instead of averaging.

**Continuous normalising flows** use the same ODE to transform a simple density into the data density. The log-density changes by $-\int\operatorname{tr}(\partial_hf_\theta)\,\mathrm dt$ (instantaneous change of variables). The probability-flow ODE of diffusion models is of this type (see [[Probability Flow ODE]]).

## References

1. R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud (2018). *Neural Ordinary Differential Equations*. NeurIPS 31. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
2. C. Rackauckas et al. (2020). *Universal Differential Equations for Scientific Machine Learning*. [arXiv:2001.04385](https://arxiv.org/abs/2001.04385)
3. A. Pal (2023). *On Efficient Training & Inference of Neural Differential Equations*. MIT thesis. [hdl.handle.net/1721.1/151379](https://hdl.handle.net/1721.1/151379)
