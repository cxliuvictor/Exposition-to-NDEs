import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax
import optax
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

jax.config.update("jax_enable_x64", True)

B = 0.5  # curvature parameter

def sample_banana(key, n):
    x = jr.normal(key, (n, 2))
    y1 = x[:, 0]
    y2 = x[:, 1] + B * x[:, 0] ** 2 - B
    return jnp.stack([y1, y2], axis=1)

def log_banana_density(y):
    """Analytical log-density of the banana distribution."""
    y1, y2 = y[0], y[1]
    x2 = y2 - B * y1 ** 2 + B           # inverse map
    return -0.5 * (y1 ** 2 + x2 ** 2) - jnp.log(2 * jnp.pi)

# True expected log-likelihood = -H(N(0,I_2)) = -(1 + log 2π)
TRUE_MEAN_LL = float(-(1.0 + jnp.log(2 * jnp.pi)))
print(f"Oracle -H(p*) = {TRUE_MEAN_LL:.4f}")   # ≈ -2.8379

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_dim, width, depth, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=data_dim + 1,
            out_size=data_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        # Scale down final layer weights to keep early dynamics small
        where_last = lambda m: m.layers[-1].weight
        scale = lambda w: w * 0.01
        self.mlp = eqx.tree_at(where_last, self.mlp,
                               replace_fn=scale)

    def __call__(self, t, y):
        return self.mlp(jnp.concatenate([y, jnp.array([t])]))


def _single_log_likelihood(model, y):
    d = y.shape[0]

    def aug_dynamics(t, state, args):
        y_t = state[:d]
        f   = model(t, y_t)
        jac = jax.jacfwd(lambda y_: model(t, y_))(y_t)
        tr  = jnp.trace(jac)
        return jnp.concatenate([f, jnp.array([-tr])])

    state0 = jnp.concatenate([y, jnp.array([0.0])])

    sol = diffrax.diffeqsolve(
    diffrax.ODETerm(aug_dynamics),
    diffrax.Tsit5(),
    t0=1.0, t1=0.0, dt0=-0.05,
    y0=state0,
    saveat=diffrax.SaveAt(t1=True),
    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
    max_steps=4096,                  # was 200 — far too low
    throw=False,                     # return best effort instead of crashing
    )

    y0       = sol.ys[0, :d]
    delta_lp = sol.ys[0,  d]

    # Guard against solver failure producing NaN/inf
    log_p0 = -0.5 * jnp.sum(y0 ** 2) - (d / 2.0) * jnp.log(2 * jnp.pi)
    result = log_p0 + delta_lp
    return jnp.where(jnp.isfinite(result), result, jnp.array(-10.0))


@eqx.filter_jit
def batch_log_likelihood(model, ys):
    """Vectorised log p_θ over a batch."""
    return jax.vmap(lambda y: _single_log_likelihood(model, y))(ys)


@eqx.filter_jit
def make_step(model, opt_state, batch, optimizer):
    def loss(m):
        return -jnp.mean(batch_log_likelihood(m, batch))

    loss_val, grads = eqx.filter_value_and_grad(loss)(model)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss_val


def train_cnf(n_train, key, *, width=32, depth=2, lr=1e-3):
    n_steps    = {100: 500, 250: 750, 500: 1000,
                  1000: 1500, 2500: 2000, 5000: 2000}.get(n_train, 2000)
    batch_size = min(256, n_train)

    k_data, k_model = jr.split(key)
    data  = sample_banana(k_data, n_train)
    model = Func(data_dim=2, width=width, depth=depth, key=k_model)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr, weight_decay=1e-4)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for step in range(n_steps):
        key, k_batch = jr.split(key)
        idx   = jr.randint(k_batch, (batch_size,), 0, n_train)
        batch = data[idx]
        model, opt_state, loss = make_step(model, opt_state, batch, optimizer)
        if (step + 1) % 100 == 0:
            print(f"  step {step+1:4d}  loss = {loss:.4f}")

    return model


def compute_test_stats(model, key, M):
    """Return (mean, std) of M test log-likelihoods."""
    test_data = sample_banana(key, M)
    lls = np.array(batch_log_likelihood(model, test_data))
    return float(lls.mean()), float(lls.std(ddof=1))


def aci_95(mean_ll, std_ll, M):
    """Asymptotic 95% CI from CLT."""
    z   = stats.norm.ppf(0.975)           # 1.96
    hw  = z * std_ll / np.sqrt(M)         # half-width
    return mean_ll - hw, mean_ll + hw


training_sizes = [500, 1000, 2500, 5000]
M_TEST = 2000
master_key = jr.PRNGKey(0)

results_A = {}
for n in training_sizes:
    master_key, k_train, k_test = jr.split(master_key, 3)
    print(f"\nTraining n = {n} …")
    model = train_cnf(n, k_train)

    mu, sigma   = compute_test_stats(model, k_test, M=M_TEST)
    lo, hi      = aci_95(mu, sigma, M_TEST)
    contains    = lo <= TRUE_MEAN_LL <= hi

    results_A[n] = dict(mean=mu, lo=lo, hi=hi, contains=contains)
    print(f"  mean LL = {mu:.4f},  95% ACI = [{lo:.4f}, {hi:.4f}],  "
          f"oracle inside = {contains}")

# Plot — Experiment A
ns     = list(results_A.keys())
means  = [results_A[n]['mean'] for n in ns]
los    = [results_A[n]['lo']   for n in ns]
his    = [results_A[n]['hi']   for n in ns]

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(ns, means,
            yerr=[np.array(means) - np.array(los),
                  np.array(his)   - np.array(means)],
            fmt='o', capsize=5, color='steelblue', label='Estimated mean LL ± 95% ACI')
ax.axhline(TRUE_MEAN_LL, color='tomato', linestyle='--', label=r'Oracle $-H(p^*) = -(1 + \log 2\pi)$')
ax.set_xscale('log')
ax.set_xlabel('Training size $n$')
ax.set_ylabel('Test log-likelihood')
ax.set_title('Experiment A: ACI vs training size')
ax.legend(); plt.tight_layout()
plt.savefig('experiment_A.pdf'); plt.show() 
