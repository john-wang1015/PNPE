import torch
import numpy as np
import pyro.distributions.transforms as T
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.utils import get_density_thresholder, RestrictedPrior
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import scipy.io as sio
from scipy.io import savemat
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class model:
    def __init__(self, dim, true_dataset, n, device='cpu'):
        self.nvars = dim
        pairs_np = np.int_(true_dataset['pairs'] - np.ones(true_dataset['pairs'].shape))
        self.pairs = torch.from_numpy(pairs_np).to(torch.long).to(device) # changed from float32 to long
        self.n = n
        self.device = device

    def run_simulation(self, theta):
        if len(theta.shape) == 1:
            theta = theta.unsqueeze(0)  
        n_theta = theta.shape[0]
        summaries = torch.zeros([n_theta, self.nvars], device=self.device)

        for i in range(n_theta):
            sims = self.simulate_GVAR(theta[i])
            summaries[i, :] = self.compute_summary(sims)

        return summaries


    def simulate_GVAR(self, theta):
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

def run_snpe():
    num_dim = 6
    low_para = -1 * torch.ones(num_dim)
    low = torch.cat([low_para, torch.zeros(1)],0)
    high = torch.ones(num_dim+1)
    prior = utils.BoxUniform(low=low, high=high) # set prior distribution

    full_dataset = sio.loadmat("data_GVAR_dim6.mat") 
    true_values = full_dataset['theta_true']
    x_0 = torch.from_numpy(full_dataset['sy']).to(torch.float32)

    sim_model = model(num_dim+1, full_dataset, 1000)

    simulators, prior = prepare_for_sbi(sim_model.run_simulation, prior)

    inference = SNPE(prior=prior, density_estimator='nsf')
    num_rounds = 3

    posteriors = []
    posterior_samples = []
    proposal = prior
    num_simulations = 10000

    for i in range(num_rounds):
        
        if i == 0:
            theta, x = simulate_for_sbi(simulators, proposal, num_simulations=23000)
        else:
            theta, x = simulate_for_sbi(simulators, proposal, num_simulations=num_simulations)
            for i in range(x.shape[0]):
                while np.isinf(x[i, :].numpy()).any() == True or np.isnan(x[i, :].numpy()).any() == True:
                    theta[i, :] = proposal.sample((1,), x_0)
                    x[i, :] = simulators(theta[i, :])
                    
        clip_val = 10  #
        clip_idx = np.unique(np.where(np.abs(x) > clip_val)[0]) 
        print(f"percentage removed: {len(clip_idx)/num_simulations}")

        theta = np.delete(theta, clip_idx, axis=0)
        x = np.delete(x, clip_idx, axis=0)
        
        density_estimator = inference.append_simulations(theta, x, proposal=proposal,).train(learning_rate=2e-4)
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        torch.save(posterior, 'svar_snpe_iter{}.pth'.format(i+1))

        posterior_sample = posterior.sample((num_simulations,), x=x_0)
        posterior_samples.append(posterior_sample)
        figure(figsize=(20, 10))
        for i in range(1,8):
            plt.subplot(2,4,i)
            sns.kdeplot(data = posterior_sample[:, i-1])
            plt.axvline(true_values[i-1])
            plt.legend(labels=['iteration', 'True'])
        plt.savefig('svar_snpe_iter{}.png'.format(i+1))
        
        proposal = posterior.set_default_x(x_0)
        
    sio.savemat("posterior_SNPE_6d.mat",{'posterior':posterior_samples})
        
    return posteriors

if __name__ == '__main__':
    run_snpe()