$$
\gdef\R{\mathbb{R}}
$$
## Notes on Kidger's NDEs talks
This is a compilation of notes on Patrick Kidger's talks across two to three seminars.
### NDEs Overview
A neural differential equation is *a differential equation (DE) that uses a neural network (NN) with learnable parameters/functions to parametrize its vector field*. A canonical starter example is a **neural ordinary differential equation** (NODE): 

$$ 
y(0) = y_0 \hspace{1cm} \frac{dy}{dt}(t) = f_\theta(t, y(t))$$ 

where $f_\theta: \R\times\R^d \to \R^d$ is some standard neural architecture, oftentimes a feed-forward one. The central idea here is that, instead of treating an ODE solver (e.g. RK4) as an external black-box that just produces numerical solutions, we now embed it inside an NN's differentiable computation graph. 

<span style="color:#888;">Remark: Traditionally, solvers only integrate and simulate the evolution of DEs with fixed parameters (e.g. of a physical/biological system), without caring how their outputs vary with respect to those parameters. They are used to generate data for an NN, which is trained afterward on these simulated data to approximate the underlying system, making it a separate pipeline from the solver. </span>

<details close>
<summary>More ground-level details about NODE</summary>
<span style="color:#887">

In an NODE, the dynamics of the ODE are defined by an NN $f_\theta$ with learnable parameters. To train this NN, it's of course essential to understand how changes in $\theta$ influence the integrated solution $y$. By recording and encompassing the solver's internal numerical steps in the differentiable computation graph, auto-differentiation can propagate gradients from the loss function back through the solver and into the NN that defines how $y$ evolves, "nudging" the parameters along the way so that the learned dynamics better match the data. This turns the solver into a trainable component of the model, enabling end-to-end learning where the system learns not just static (plain "input $\to$ output") mappings, but the continuous-time evolution described by the ODE itself.

Note that the ODE describes the evolution of $y$, which lives in a *latent space* (some internal representation of the raw input space). The solver integrates this NODE from $y(0)$ (the encoded input) to $y(T)$, then passes $y(T)$ through affine and softmax layers to get output/predictions and compute a loss.

</span>
</details>
...

...

It's worth mentioning that this seemingly complicated nesting of an NN inside an DE is nothing groundbreaking within the mathematics and ML communities. For example, the SIR model in epidemiology is as follows:

$$
\frac{d}{dt} \begin{pmatrix} s(t) \\ i(t) \\ r(t) \end{pmatrix} = \begin{pmatrix} -b s(t)i(t) \\ b s(t)i(t) - k i(t) \\ k i(t) \end{pmatrix}. 
$$

Both NODEs and the SIR model are parametrized DEs with some theoretical models for the RHS, with the mere difference being that SIR is specialized to fit a specific phenomenon (namely the evolution of a disease), while an NODE is a generalized "fitter" applicable to lots of things thanks to the NN architecture. In fact, the SIR model can transform into an NODE by making the parameters $b$ and $k$ learnable.

In the framework of deep learning (DL), we may also see a shadow of NODE: by applying Euler method on $\frac{dy}{dt}(t) = f_\theta(t, y(t))$, we obtain a residual network (ResNet): 

$$ 
y_{t_{n+1}} = y_{t_{n}} + \Delta tf_\theta(t_n, y_{t_{n}}).
$$

Interestingly, this connection between DL architectures and NDEs is ubiquitous: 
1. ResNet $\leftrightarrow$ neural ordinary differential equations (NODEs)
2. GRUs/LSTMs $\leftrightarrow$ neural controlled differential equations (NCDEs)
3. StyleGAN2 $\leftrightarrow$ neural stochastic differential equations (NSDEs)

NDEs have a myriad applications in real life, often demonstrating state-of-the-art performance in areas like modelling (physical, biological, financial, etc), regular/irregular time series analysis, generative modelling, and many more. 

NDEs offer practically a best-of-both-world approach: the NN side offers high-capacity function approximation and high trainability, while the DE side offers strong prior on model space, memory-efficiency, and theoretical understanding via a well-understood literature.


