import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import stats
import qmcpy
import normflows as nf
import torch
from tqdm import tqdm
import customdistributions as cd
torch.set_default_tensor_type(torch.DoubleTensor)

K = 16
torch.manual_seed(2026)

latent_size = 2
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
for i in range(K):
    s = nf.nets.MLP([latent_size,2*latent_size, latent_size], init_zeros=True)
    t = nf.nets.MLP([latent_size, 2*latent_size, latent_size], init_zeros=True)
    acb = nf.nets.MLP([1, 32*latent_size, 32*latent_size, latent_size], init_zeros=True)
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    #flows += [nf.flows.AffineCouplingBlock(acb)]
    #flows += [nf.flows.BatchNorm()]
    flows += [nf.flows.ActNorm(latent_size)]
    #flows += [nf.flows.Permute(latent_size,mode = "swap")]

# Set target and q0
target = cd.MultivariateStudentT(torch.tensor([0,0],dtype = torch.float64),torch.tensor([[1,1],[1,2]],dtype=torch.float64),20)
q0 = cd.Multivariate_Diag_t_Torch([0,0],[1.0,2.0],20)

# Construct flow model
nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)

# Move model on GPU if available
enable_cuda = False
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
print
nfm = nfm.to(device)
nfm = nfm.double()
#
plotsize = 5
# Initialize ActNorm
z, _ = nfm.sample(num_samples=2 ** 9)
z_np = z.to('cpu').data.numpy()
plt.figure(figsize=(15, 15))
plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (200, 200), range=[[-plotsize, plotsize], [-plotsize, plotsize]])
plt.gca().set_aspect('equal', 'box')
plt.savefig(".//Visualization//init_guess.png",dpi = 150)
plt.close()

# Plot target distribution
grid_size = 200
xx, yy = torch.meshgrid(torch.linspace(-plotsize, plotsize, grid_size), torch.linspace(-plotsize, plotsize, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.double().to(device)
log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
prob_target = torch.exp(log_prob)

# Plot initial posterior distribution
log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(15, 15))
plt.pcolormesh(xx, yy, prob.data.numpy())
plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
plt.gca().set_aspect('equal', 'box')
plt.savefig(".//Visualization//init_distributions.png",dpi = 150)
plt.close()

# Train model
max_iter = 10000
num_samples = 2 ** 9
anneal_iter = 100
annealing = False
show_iter = 500


loss_hist = np.array([])

optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max_iter,1e-07)
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    if annealing:
        loss = nfm.reverse_kld(num_samples, beta=np.min([1., 0.01 + it / anneal_iter]))
    else:
        #loss = nfm.reverse_alpha_div(num_samples, dreg=True, alpha=0.5)
        loss = nfm.reverse_kld(num_samples)
        #x = target.sample(num_samples).to(device)
        #loss = nfm.forward_kld(x)
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    #loss = torch.abs(loss)
    """
    for group in optimizer.param_groups:
        if loss<0:
            group['maximize']=True
        else:
            group['maximize']=False
    """
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
        #scheduler.step()
    else:
        optimizer = torch.optim.SGD(nfm.parameters(), lr=1e-6)
        #scheduler.step(epoch = it)
        print("Loss untypical")
    
    
    # Plot learned posterior
    if (it + 1) % show_iter == 0:
        nfm.eval()
        log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
        nfm.train()
        prob = torch.exp(log_prob)
        prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.data.numpy())
        plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
        plt.contour(xx, yy, prob.data.numpy(), cmap=plt.get_cmap('pink'), linewidths=2)
        plt.gca().set_aspect('equal', 'box')
        plt.savefig(f".//Visualization//training_OldLoss_{it}.png",dpi = 150)
        plt.close()


        plt.figure(figsize=(10, 10))
        plt.plot(loss_hist, label='loss')
        plt.legend()
        plt.savefig(".//Visualization//loss_OldLoss.png",dpi = 150)
        plt.close()
# Plot learned posterior distribution
log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(15, 15))
plt.pcolormesh(xx, yy, prob.data.numpy())
plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
plt.gca().set_aspect('equal', 'box')
plt.savefig(".//Visualization//end_result_OldLoss.png",dpi = 150)
plt.close()