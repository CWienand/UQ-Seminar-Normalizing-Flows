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
import network_architectures as na
import sklearn
from sklearn.datasets import make_spd_matrix


torch.set_default_dtype(torch.float64)
SCIPY_ARRAY_API=1


K = 16
torch.manual_seed(2026)
latent_size = 2

testcasename = f"{latent_size}D-MC-QMC-Compare"

target_space = torch.tensor(make_spd_matrix(n_dim=latent_size))
print(target_space)
input_space = torch.diagonal(target_space)
location = torch.zeros_like(input_space)

input_representation = sklearn.manifold.TSNE(n_components = 2,perplexity=30.0)
target_representation = sklearn.manifold.TSNE(n_components = 2,perplexity=30.0)


b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
for i in range(K):
    s = na.CustomMLP([latent_size,32*latent_size,16*latent_size, latent_size], init_zeros=True)
    t = na.CustomMLP([latent_size,32*latent_size,16*latent_size, latent_size], init_zeros=True)
    acb = na.CustomMLP([1, 32*latent_size, 32*latent_size, latent_size], init_zeros=True)
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    #flows += [nf.flows.AffineCouplingBlock(acb)]
    #flows += [nf.flows.BatchNorm()]
    flows += [nf.flows.ActNorm(latent_size)]
    #flows += [nf.flows.Permute(latent_size,mode = "swap")]

# Set target and q0
target = cd.MultivariateStudentT(location,target_space,20)
q0 = cd.Multivariate_Diag_Base_MC_mapping(location,input_space,20)
q1 = cd.Multivariate_Diag_t_Torch(location,input_space,20)
q2 = cd.Multivariate_Diag_t_qmc(location,input_space,20)

# Construct flow model
nfm0 = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
nfm1 = nf.NormalizingFlow(q0=q1, flows=flows, p=target)
nfm2 = nf.NormalizingFlow(q0=q2, flows=flows, p=target)

#
plotsize = 5
# Initialize ActNorm
z, _ = nfm0.sample(num_samples=2 ** 9)
z_np = z.to('cpu').data.numpy()
z, _ = nfm1.sample(num_samples=2 ** 9)
z_np = z.to('cpu').data.numpy()
z, _ = nfm2.sample(num_samples=2 ** 9)
z_np = z.to('cpu').data.numpy()

#plt.figure(figsize=(15, 15))
#plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (200, 200), range=[[-plotsize, plotsize], [-plotsize, plotsize]])
#plt.gca().set_aspect('equal', 'box')
#plt.savefig(f".//Visualization//init_guess_{testcasename}.png",dpi = 150)
#plt.close()

# Plot target distribution
#grid_size = 200
#xx, yy = torch.meshgrid(torch.linspace(-plotsize, plotsize, grid_size), torch.linspace(-plotsize, plotsize, grid_size))
#zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
#zz = zz.double().to(device)
#log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
#prob_target = torch.exp(log_prob)

# Plot initial posterior distribution
#log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
#prob = torch.exp(log_prob)
#prob[torch.isnan(prob)] = 0

#plt.figure(figsize=(15, 15))
#plt.pcolormesh(xx, yy, prob.data.numpy())
#plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
#plt.gca().set_aspect('equal', 'box')
#plt.savefig(f".//Visualization//init_distributions_{testcasename}.png",dpi = 150)
#plt.close()

# Train model
max_iter = 5000
num_samples = 2 ** 10
anneal_iter = 1000
annealing = False
show_iter = 1000


loss_hist0 = np.array([])
loss_hist1 = np.array([])
loss_hist2 = np.array([])

optimizer0 = torch.optim.SGD(nfm0.parameters(), lr=5e-5,weight_decay=1e-06)
scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer0,max_iter,1e-06)
optimizer1 = torch.optim.SGD(nfm0.parameters(), lr=5e-5,weight_decay=1e-06)
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1,max_iter,1e-06)
optimizer2 = torch.optim.SGD(nfm0.parameters(), lr=5e-5,weight_decay=1e-06)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2,max_iter,1e-06)
for it in tqdm(range(max_iter)):
    optimizer0.zero_grad()
    #optimizer1.zero_grad()
    #optimizer2.zero_grad()
    if annealing:
        pass
        loss = nfm0.reverse_kld(num_samples, beta=np.min([1., 0.01 + it / anneal_iter]))
    else:
        #loss = nfm.reverse_alpha_div(num_samples, dreg=True)
        loss0 = nfm0.reverse_kld(num_samples)
        #loss1 = nfm1.reverse_kld(num_samples)
        #loss2 = nfm2.reverse_kld(num_samples)
        #x = target.sample(num_samples).to(device)
        #loss = nfm.forward_kld(x)
    loss_hist0 = np.append(loss_hist0, loss0.to('cpu').data.numpy())
    #loss_hist1 = np.append(loss_hist1, loss1.to('cpu').data.numpy())
    #loss_hist2 = np.append(loss_hist2, loss2.to('cpu').data.numpy())
    #loss = torch.abs(loss)
    """
    for group in optimizer.param_groups:
        if loss<0:
            group['maximize']=True
        else:
            group['maximize']=False
    """
    if ~(torch.isnan(loss0) | torch.isinf(loss0)):
        loss0.backward()
        optimizer0.step()
        scheduler0.step()
        #loss1.backward()
        #optimizer1.step()
        #scheduler1.step()
        #loss2.backward()
        #optimizer2.step()
        #scheduler2.step()
    else:
        #optimizer = torch.optim.SGD(nfm.parameters(), lr=1e-6)
        #scheduler.step(epoch = it)
        print("Loss untypical")
    
for it in tqdm(range(max_iter)):
    #optimizer0.zero_grad()
    optimizer1.zero_grad()
    #optimizer2.zero_grad()
    if annealing:
        pass
        loss = nfm0.reverse_kld(num_samples, beta=np.min([1., 0.01 + it / anneal_iter]))
    else:
        #loss = nfm.reverse_alpha_div(num_samples, dreg=True)
        #loss0 = nfm0.reverse_kld(num_samples)
        loss1 = nfm1.reverse_kld(num_samples)
        #loss2 = nfm2.reverse_kld(num_samples)
        #x = target.sample(num_samples).to(device)
        #loss = nfm.forward_kld(x)
    #loss_hist0 = np.append(loss_hist0, loss0.to('cpu').data.numpy())
    loss_hist1 = np.append(loss_hist1, loss1.to('cpu').data.numpy())
    #loss_hist2 = np.append(loss_hist2, loss2.to('cpu').data.numpy())
    #loss = torch.abs(loss)
    """
    for group in optimizer.param_groups:
        if loss<0:
            group['maximize']=True
        else:
            group['maximize']=False
    """
    if ~(torch.isnan(loss0) | torch.isinf(loss0)):
        #loss0.backward()
        #optimizer0.step()
        #scheduler0.step()
        loss1.backward()
        optimizer1.step()
        scheduler1.step()
        #loss2.backward()
        #optimizer2.step()
        #scheduler2.step()
    else:
        #optimizer = torch.optim.SGD(nfm.parameters(), lr=1e-6)
        #scheduler.step(epoch = it)
        print("Loss untypical")
for it in tqdm(range(max_iter)):
    #optimizer0.zero_grad()
    #optimizer1.zero_grad()
    optimizer2.zero_grad()
    if annealing:
        pass
        loss = nfm0.reverse_kld(num_samples, beta=np.min([1., 0.01 + it / anneal_iter]))
    else:
        #loss = nfm.reverse_alpha_div(num_samples, dreg=True)
        loss0 = nfm0.reverse_kld(num_samples)
        #loss1 = nfm1.reverse_kld(num_samples)
        loss2 = nfm2.reverse_kld(num_samples)
        #x = target.sample(num_samples).to(device)
        #loss = nfm.forward_kld(x)
    #loss_hist0 = np.append(loss_hist0, loss0.to('cpu').data.numpy())
    #loss_hist1 = np.append(loss_hist1, loss1.to('cpu').data.numpy())
    loss_hist2 = np.append(loss_hist2, loss2.to('cpu').data.numpy())
    #loss = torch.abs(loss)
    """
    for group in optimizer.param_groups:
        if loss<0:
            group['maximize']=True
        else:
            group['maximize']=False
    """
    if ~(torch.isnan(loss0) | torch.isinf(loss0)):
        #loss0.backward()
        #optimizer0.step()
        #scheduler0.step()
        #loss1.backward()
        #optimizer1.step()
        #scheduler1.step()
        loss2.backward()
        optimizer2.step()
        scheduler2.step()
    else:
        #optimizer = torch.optim.SGD(nfm.parameters(), lr=1e-6)
        #scheduler.step(epoch = it)
        print("Loss untypical")
    
    # Plot learned posterior
    #if (it + 1) % show_iter == 0:
        #nfm.eval()
        #log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
        #samples, _ = nfm.sample(2**7)
        #nfm.train()
        #prob = torch.exp(log_prob)
        #prob[torch.isnan(prob)] = 0

        #plt.figure(figsize=(15, 15))
        #plt.pcolormesh(xx, yy, prob.data.numpy())
        #plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
        #plt.contour(xx, yy, prob.data.numpy(), cmap=plt.get_cmap('pink'), linewidths=2)
        #plt.scatter(samples[:,0].detach().numpy(),samples[:,1].detach().numpy(),color="w")
        #plt.gca().set_aspect('equal', 'box')
        #plt.savefig(f".//Visualization//training_{testcasename}_{it}.png",dpi = 150)
        #plt.close()


plt.figure(figsize=(20, 10))
#plt.yscale("log")
plt.plot(loss_hist0, label='loss_MC_mapping')
plt.plot(loss_hist1, label='loss_MC')
plt.plot(loss_hist2, label='loss_QMC_mapping')
plt.legend()
plt.savefig(f".//Visualization//loss_{testcasename}.png",dpi = 300)
plt.close()
plt.figure(figsize=(20, 10))
plt.yscale("log")
plt.plot(loss_hist0, label='loss_MC_mapping')
plt.plot(loss_hist1, label='loss_MC')
plt.plot(loss_hist2, label='loss_QMC_mapping')
plt.legend()
plt.savefig(f".//Visualization//loss_log_{testcasename}.png",dpi = 300)
plt.close()
z, _ = nfm0.sample(num_samples=2 ** 9)
z_np = z.to('cpu').data.numpy()

z_np_tsne = target_representation.fit_transform(z_np)
print(z_np_tsne.shape)

plt.scatter(z_np_tsne[:,0],z_np_tsne[:,1])
plt.show()
# Plot learned posterior distribution
#log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
#prob = torch.exp(log_prob)
#prob[torch.isnan(prob)] = 0

#plt.figure(figsize=(15, 15))
#plt.pcolormesh(xx, yy, prob.data.numpy())
#plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
#plt.gca().set_aspect('equal', 'box')
#plt.savefig(f".//Visualization//end_result_{testcasename}.png",dpi = 150)
#plt.close()