import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import stats
import qmcpy
import normflows as nf
import torch
from tqdm import tqdm

def t_student_pdf(x, nu_IS, sigma_IS):
  """computes the pdf of the univariate t-student distribution

  Args:
      x (float): input value
      nu_IS (float): degrees of freedom
      sigma_IS (float): scale parameter

  Returns:
      float: pdf evaluated at x
  """
  return stats.t.pdf( x =  x, df = nu_IS, loc = 0, scale = sigma_IS)

def t_student_ppf(x, nu_IS, sigma_IS):
  """computes the inverse cumulative distribution function of the univariate t-student distribution

  Args:
      x (float):  input value
      nu_IS (float): degrees of freedom
      sigma_IS (float): scale parameter

  Returns:
      float: inverse cumulative distribution function evaluated at x
  """
  return stats.t.ppf( q =  x, df = nu_IS, loc = 0, scale = sigma_IS)


def covariance_matrix(sigma, rho):
    """Compute the covariance matrix.
    Args:
    - sigma (array): Array of volatilities of each stock.
    - rho (array): Correlation matrix.
    Returns:
    - SIGMA (array): Covariance matrix.
    """
    sigma = np.diag(sigma)  # Diagonal matrix of volatilities
    SIGMA = np.dot(sigma, np.dot(rho, sigma))  # Covariance matrix calculation
    return SIGMA

def VG_characteristic_function(u, SIGMA, T, r, theta, nu):
    """Calculate the characteristic function of Variance-Gamma process.
    Args:
    - u (array): Vector in Rd.
    - SIGMA (array): Covariance matrix.
    - T (float): Terminal time.
    - r (float): Short rate.
    - theta (array): Array of theta values.
    - nu (float): Nu parameter.
    Returns:
    - phi (complex): Characteristic function value.
    """
    w = (1/nu) * np.log(1 - nu * theta - 0.5 * nu * np.diag(SIGMA))  # Martingale correction term
    phi = np.exp(np.multiply(1j * T, np.dot(r + w, u))) * (1 - np.multiply(1j * nu, np.dot(theta, u)) +
                                                           0.5 * nu * np.dot(u, np.dot(SIGMA, u))) ** (-T/nu)
    return phi

def fourier_payoff_call_on_min(u):
    """Compute the Fourier of the payoff of scaled (K = 1) call on min option.
    Args:
    - u (array): Array of Fourier frequencies.
    Returns:
    - payoff (float): Call on min option payoff Fourier transofrm value.
    """
    denominator = (np.multiply(1j, np.sum(u)) - 1) * np.prod(np.multiply(1j, u))
    return 1 / denominator

def fourier_payoff_basket_put(u):
  """Compute the Fourier of the payoff of scaled (K = 1) basket put option.
  Args:
  - u (array): Array of Fourier frequencies.
  Returns:
  - payoff (float): Call on min option payoff Fourier transofrm value.
  """
  numerator = np.prod(sc.special.gamma(np.multiply(-1j,u)))
  denominator = sc.special.gamma(-1j*(np.sum(u))+2)
  return (numerator/denominator)

pdf = t_student_pdf(np.linspace(-10,10,100),1,1)
plt.plot(np.linspace(-10,10,100),pdf)
plt.show()
plt.close()

base = nf.distributions.base.Uniform(2,low=-1.0,high=1.0)
#base = nf.distributions.base.UniformGaussian(2,1)
#base = nf.distributions.base.DiagGaussian(2)
# Define list of flows
num_layers = 32
flows = []
for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))

# If the target density is given
target = nf.distributions.target.TwoMoons()
model = nf.NormalizingFlow(base, flows, target)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)

# Plot target distribution
grid_size = 200
xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)

log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(15, 15))
plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
plt.gca().set_aspect('equal', 'box')
plt.show()

# Plot initial flow distribution
model.eval()
log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(15, 15))
plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
plt.gca().set_aspect('equal', 'box')
plt.show()

# Train model
max_iter = 1000
num_samples = 2 ** 9
show_iter = 500


loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)

for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
    x = target.sample(num_samples).to(device)
    
    # Compute loss
    loss = model.forward_kld(x)
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
    # Plot learned distribution
    if (it + 1) % show_iter == 0:
        model.eval()
        log_prob = model.log_prob(zz)
        model.train()
        prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
        prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
        plt.gca().set_aspect('equal', 'box')
        plt.show()

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

# Plot target distribution
f, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 7))

log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

ax[0].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')

ax[0].set_aspect('equal', 'box')
ax[0].set_axis_off()
ax[0].set_title('Target', fontsize=24)

# Plot learned distribution
model.eval()
log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

ax[1].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')

ax[1].set_aspect('equal', 'box')
ax[1].set_axis_off()
ax[1].set_title('Real NVP', fontsize=24)

plt.subplots_adjust(wspace=0.1)

plt.show()