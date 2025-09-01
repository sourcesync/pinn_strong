
import os
import time

import jax
import jax.numpy as jnp
from jax import jit, vmap, jacrev
from jax.tree_util import tree_map

import ml_collections
from absl import logging
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from jaxpi.samplers import UniformSampler
from jaxpi.utils import save_checkpoint, restore_checkpoint, flatten_pytree, jacobian_fn

import models
from utils import get_dataset

from configs import plain

# Import config
config = plain.get_config()

# Problem setup
T = 2.0  # final time
L = 2 * jnp.pi  # length of the domain
c = 50  # advection speed
n_t = 200  # number of time steps
n_x = 128  # number of spatial points

# Get  dataset
u_ref, t_star, x_star = get_dataset(T, L, c, n_t, n_x)
u0 = u_ref[0, :]


# Define domain
t0 = t_star[0]
t1 = t_star[-1]

x0 = x_star[0]
x1 = x_star[-1]

dom = jnp.array([[t0, t1],
                 [x0, x1]])

# Initialize residual sampler
res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))


# Initialize model
model = models.Advection(config, u0, t_star, x_star, c)


# Restore model paramteres
workdir = os.getcwd()
ckpt_path = os.path.join(workdir, 'plain', 'ckpt')
model.state = restore_checkpoint(model.state, ckpt_path)
params = model.state.params

# Compute L2 error
l2_error = model.compute_l2_error(params, u_ref)
print('L2 error: {:.3e}'.format(l2_error))

# Prediction
u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
TT, XX = jnp.meshgrid(t_star, x_star, indexing='ij')

# Plot results
fig = plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.pcolor(TT, XX, u_ref, cmap='jet')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title('Exact')
plt.tight_layout()

plt.subplot(1, 3, 2)
plt.pcolor(TT, XX, u_pred, cmap='jet')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title('Predicted')
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.pcolor(TT, XX, jnp.abs(u_ref - u_pred), cmap='jet')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title('Absolute error')
plt.tight_layout()

fig.savefig("./adv.png")
print("Wrote adv.png")
