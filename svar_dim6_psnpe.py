import torch
import numpy as np

import pyro.distributions as dist
import pyro.distributions.transforms as T

import sbi
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import scipy.io as sio
from scipy.io import savemat
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
_ = torch.manual_seed(42)

class model:
    def __init__(self, dim, true_dataset, n, device='cpu'):
        self.nvars = dim
        # NOTE: RYAN CHANGE PAIRS
        pairs_np = np.int_(true_dataset['pairs'] - np.ones(true_dataset['pairs'].shape))
        self.pairs = torch.from_numpy(pairs_np).to(torch.long).to(device) # changed from float32 to long
        self.n = n
        self.device = device

    def run_simulation(self, theta):
        # theta = theta.to(self.device)
        if len(theta.shape) == 1:
            theta = theta.unsqueeze(0)  # Reshape to 2D tensor
        n_theta = theta.shape[0]
        summaries = torch.zeros([n_theta, self.nvars], device=self.device)

        for i in range(n_theta):
            sims = self.simulate_GVAR(theta[i])
            summaries[i, :] = self.compute_summary(sims)

        return summaries


    def simulate_GVAR(self, theta):
        # theta = theta.to(self.device)
        X = -0.1 * torch.eye(self.nvars - 1, device=self.device)

        for ii, pair in enumerate(self.pairs):
            X[pair[0], pair[1]] = theta[ii*2]
            X[pair[1], pair[0]] = theta[ii*2+1]

        sigma = theta[-1]
        Y0 = torch.normal(mean=0.0, std=abs(sigma), size=(self.nvars-1,), device=self.device)
        Y = torch.zeros(self.nvars-1, self.n, device=self.device)

        for t in range(1, self.n):
            Y[:, t] = torch.matmul(X, Y0 if t == 1 else Y[:, t-1]) + \
                      torch.normal(mean=0.0, std=abs(sigma), size=(self.nvars - 1,), device=self.device)
        return Y

    def compute_summary(self, Y):
        S = torch.zeros(self.nvars, dtype=torch.float32, device=self.device)

        for ii, pair in enumerate(self.pairs):
            S[ii*2] = torch.mean(Y[pair[0], 1:] * Y[pair[1], 0:-1])
            S[ii*2+1] = torch.mean(Y[pair[1], 1:] * Y[pair[0], 0:-1])

        S[-1] = torch.std(Y)

        return S
    
def standarized_IQR(data):
    standardized_IQR_data = torch.zeros(data.shape)
    IQR = torch.zeros(data.shape[1])
    median = torch.zeros(data.shape[1])

    for i in range(data.shape[1]):
        column = data[:, i]

        median[i] = torch.median(column)

        q1 = torch.quantile(column, 0.25)
        q3 = torch.quantile(column, 0.75)

        IQR[i] = q3 - q1

        standardized_column = (column - median[i]) / IQR[i]

        standardized_IQR_data[:, i] = standardized_column

    return standardized_IQR_data, median, IQR

def inv_standarized_IQR(data, medians, iqrs):
    original_data = torch.zeros(data.shape)

    for i in range(data.shape[1]):
        standardized_column = data[:, i]
        median = medians[i]
        iqr = iqrs[i]

        original_column = (standardized_column * iqr) + median
        original_data[:, i] = original_column

    return original_data


def logit_transform(data, lower_bounds, upper_bounds):
    data, lower_bounds, upper_bounds = data.cpu(), lower_bounds.cpu(), upper_bounds.cpu()
    n, nvar = data.shape
    trans_data = torch.zeros([n, nvar])

    for i in range(n):
        num = data[i, :] - lower_bounds
        denom = upper_bounds - data[i, :]
        trans_data[i, :] = torch.log(num / denom)

    return trans_data.cuda()

def inverse_logit_transform(data, lower_bounds, upper_bounds):
    data, lower_bounds, upper_bounds = data.cpu(), lower_bounds.cpu(), upper_bounds.cpu()
    n, nvar = data.shape
    trans_inv_data = torch.zeros([n, nvar])

    for i in range(n):
        num = torch.exp(data[i, :]) + lower_bounds
        denom = 1 + torch.exp(data[i, :])
        trans_inv_data[i, :] = num / denom

    return trans_inv_data.cuda()

class CustomPrior:
    def __init__(self, data, upper, lower, flow_dist):
        self.upper = upper
        self.lower = lower
        self.flow_dist = flow_dist

    def sample(self, size):
        return self.flow_dist.sample(size)

    def log_prob(self, data):
        log_probs = torch.mean(self.flow_dist.log_prob(data), dim=1)
            
        return log_probs

def run_psnpe():
    num_dim = 6
    low_para = -1 * torch.ones(num_dim)
    low = torch.cat([low_para, torch.zeros(1)],0)
    high = torch.ones(num_dim+1)
    prior = utils.BoxUniform(low=low, high=high) # set prior distribution

    full_dataset = sio.loadmat("data_GVAR_dim6.mat") 
    true_values = full_dataset['theta_true']
    x_0 = torch.from_numpy(full_dataset['sy']).to(torch.float32)

    part_vals_new = sio.loadmat("results_summ_pilot.mat")['part_vals_smc']#trans_finv(part_vals,low,high)
    part_sim = sio.loadmat("results_summ_pilot.mat")['part_sim_smc']

    theta = torch.from_numpy(part_vals_new).to(torch.float32)
    x = torch.from_numpy(part_sim).to(torch.float32)
    d_param = 7

    base_dist = dist.Normal(torch.zeros(d_param), torch.ones(d_param))
    spline_transform = T.Spline(d_param, count_bins=64)
    flow_dist = dist.TransformedDistribution(base_dist, [spline_transform])

    steps = 10000
    optimizer = torch.optim.Adam(spline_transform.parameters(), lr=1e-4)
    for step in range(steps):
        optimizer.zero_grad()
        loss = -flow_dist.log_prob(theta).mean()
        loss.backward()
        optimizer.step()
        flow_dist.clear_cache()

        if step % 1000 == 0:
            print('step: {}, loss: {}'.format(step, loss.item()))
            
    #samples_inv_iqr = inv_standarized_IQR(flow_dist.sample([10000]), theta_median, theta_iqr)
    #data = inverse_logit_transform(samples_inv_iqr, low, high).cpu()

    data = flow_dist.sample([10000]).cpu()

    figure(figsize=(20, 10))
    for i in range(1,8):
        plt.subplot(2,4,i)
        sns.kdeplot(data = data[:, i-1])
        plt.axvline(true_values[i-1])
        plt.legend(labels=['iteration', 'True'])
    plt.savefig('svar_psnpe_preconditioning2.png')
    
    sim_model = model(7, full_dataset, 1000)
    
    c_prior = CustomPrior(theta, high, low, flow_dist)
    simulator, prior = prepare_for_sbi(sim_model.run_simulation, c_prior)
    inference_smc = SNPE_C(prior=prior, density_estimator='nsf')

    num_rounds = 2

    posteriors_smc = []
    posterior_samples_smc = []
    proposal_smc = prior

    num_simulations = 10000

    for i in range(num_rounds):
        if i == 0:
            #theta_trans = prior.sample([num_simulations])
            #theta = inverse_logit_transform(inv_standarized_IQR(theta_trans, 
            #                                                    theta_median, theta_iqr), 
            #                                low, high).cpu()
            #x = simulator(theta).cpu()
            theta = torch.from_numpy(part_vals_new).to(torch.float32)
            x = torch.from_numpy(part_sim).to(torch.float32)
        else:
            theta, x = simulate_for_sbi(simulator, proposal_smc, num_simulations=num_simulations)
        
        nan_idx = np.unique(np.where(np.isnan(x))[0])  
        extreme_val_idx = np.unique(np.where(np.isinf(x))[0])
        remove_idx = np.unique(np.concatenate((nan_idx, extreme_val_idx)))

        print(f"Percentage removed: {len(remove_idx) / num_simulations * 100:.2f}%")

        theta = np.delete(theta, remove_idx, axis=0)
        x = np.delete(x, remove_idx, axis=0)
        
        density_estimator_smc = inference_smc.append_simulations(
            theta, x, proposal = proposal_smc
        ).train()
            
        posterior_smc = inference_smc.build_posterior(density_estimator_smc)
        posterior_sample_smc = posterior_smc.sample((num_simulations,), x=x_0)
        torch.save(posteriors_smc, 'svar_psnpe_iter{}.pth'.format(i+1))
        
        figure(figsize=(20, 10))
        for j in range(1,22):
            plt.subplot(4,6,j)
            sns.kdeplot(data = posterior_sample_smc[:, j-1])
            plt.axvline(true_values[j-1])
            plt.legend(labels=['iteration', 'True'])
        plt.savefig('svar_psnpe_iter{}.png'.format(i+1))

        posterior_samples_smc.append(posterior_sample_smc)
        posteriors_smc.append(posterior_smc)
        proposal_smc = posterior_smc.set_default_x(x_0)
        
    sio.savemat("posterior_SNPE_6d.mat",{'posterior':posterior_sample_smc})
    
    return posteriors_smc

if __name__ == '__main__':
    run_psnpe()