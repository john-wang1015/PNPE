import torch
import numpy as np
import matlab.engine

import pyro.distributions as dist
import pyro.distributions.transforms as T
from torch.distributions import Transform, TransformedDistribution, MultivariateNormal, Normal, Uniform

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

def model(theta):
    eng = matlab.engine.start_matlab()
    bvcbm_path = os.path.abspath('PNPE')
    model_path = os.path.join(bvcbm_path, 'Model')
    eng.addpath(bvcbm_path, nargout=0)
    eng.addpath(model_path, nargout=0)  
    print(theta.shape)
    theta_matlab = matlab.double(theta.tolist())

    sx_all = np.zeros((len(theta.tolist()), 19))
    print(sx_all.shape)
    sx_all = eng.simulator_pal(theta_matlab, 19,  nargout=1)
    eng.quit()
    sx_all = np.asarray(sx_all)
    print(sx_all)
    return torch.from_numpy(sx_all).to(torch.float32)

def simulators(theta):
    return model(theta)

    
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
    
class adaptive_smc_abc():
    def __init__(self, obs, simulator, num_params, lower_bounds, upper_bounds):
        self.device = torch.device("cpu")

        self.N = 1000
        self.a = 0.5
        self.c = 0.01
        self.p_acc_min = 0.1
        self.num_params = num_params
        
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
        self.obs = obs
        
        self.simulator = simulator

    # define distance function
    def _dist_func(self, sim):
        return torch.sum((self.part_obs-sim)**2)

    def _trans_f(self, theta):
        # Perform the logit transformation
        transformed = torch.log((theta - self.lower_bounds) / (self.upper_bounds - theta))
        
        return transformed
    
    def _prior_sampler(self):
        return Uniform(self.lower_bounds, self.upper_bounds).sample().to(self.device)
    
    def _prior_pdf(theta_trans):
        exp_term = torch.exp(theta_trans)
        pdf_vals = exp_term / (1 + exp_term) ** 2
        pdf_product = torch.prod(pdf_vals, dim=0)  
        return pdf_product

    def _trans_finv(self, theta_trans):
        exp_theta = torch.exp(theta_trans)
        transformed = (self.upper_bounds * exp_theta + self.lower_bounds) / (1 + exp_theta)
        return transformed
    
    def _initial_run(self):
        self.part_obs = self.obs

        # values for adaptive steps
        self.num_drop = int(np.floor(self.N * self.a))
        self.num_keep = self.N-self.num_drop
        self.mcmc_trials = 5

        # Initialise particle data structures
        part_vals = torch.zeros(self.N,self.num_params)
        part_s = torch.zeros(self.N,1)
        part_sim = torch.zeros(self.N, self.part_obs.shape[0])

        for i in range(self.N):
            part_vals[i,:] = self._prior_sampler()
            part_sim[i,:] = self.simulator(part_vals[i,:])
            part_s[i,:] = self._dist_func(part_sim[i,:])

        part_vals = part_vals.reshape([self.N,self.num_params])
        part_sim = part_sim.reshape([self.N, self.part_obs.shape[0]])
        part_s = part_s.reshape([1,self.N])
            
        self.sims = self.N
        dist_final = 0
        self.dist_history = max(part_s)
        self.sims_history = self.N

        for i in range(self.N):
            part_vals[i,:] = self._trans_f(part_vals[i,:])

        part_s, ix = torch.sort(part_s)
        part_vals = part_vals[ix, :].reshape([self.N,self.num_params])
        part_sim = part_sim[ix, :].reshape([self.N, self.part_obs.shape[0]])

        dist_max = part_s[0,self.N-1].to(self.device)
        dist_next = part_s[0,self.num_keep-1].to(self.device)
        print(dist_max,dist_next,dist_final)
        self.dist_t = dist_next
        self.p_acc_t = 0
        
        return part_vals, part_sim, dist_max,dist_next,dist_final

    def run_smc_abc(self):
        part_vals, part_sim, dist_max,dist_next,dist_final = self._initial_run()
        
        while dist_max > dist_final:
            cov_matrix = (2.38**2) * torch.cov(part_vals[0:self.num_keep, :].t()) / self.num_params
            r = torch.multinomial(torch.ones(self.num_keep), self.N - self.num_keep, replacement=True)
            part_vals[self.num_keep:self.N, :] = part_vals[r, :]
            part_s[0,self.num_keep:self.N] = part_s[0,r]
            part_sim[self.num_keep:self.N, :] = part_sim[r, :]

            i_acc = torch.zeros(self.N - self.num_keep, 1)
            sims_mcmc = torch.zeros(self.N - self.num_keep, 1)

            for i in range(self.num_keep, self.N):
                for _ in range(self.mcmc_trials):
                    dist = MultivariateNormal(part_vals[i, :], cov_matrix)
                    part_vals_prop = dist.sample()

                    prior_curr = self._prior_pdf(part_vals[i, :])
                    prior_prop = self._prior_pdf(part_vals_prop)

                    if torch.isnan(prior_prop / prior_curr) or torch.rand(1) > prior_prop / prior_curr:
                        continue

                    prop = self._trans_finv(part_vals_prop)
                    part_sim_prop = self.simulator(prop)
                    dist_prop = self._dist_func(part_sim_prop)
                    sims_mcmc[i - self.num_keep] += 1

                    if dist_prop <= dist_next:
                        part_vals[i, :] = part_vals_prop
                        part_s[0,i] = dist_prop
                        part_sim[i, :] = part_sim_prop
                        i_acc[i - self.num_keep] += 1

            acc_rate = sum(i_acc) / (self.mcmc_trials * (self.N - self.num_keep))
            mcmc_iters = int(np.floor(np.log(self.c) / np.log(1 - acc_rate) + 1))
            print(f'Total number of mcmc moves for current target is {mcmc_iters}, number remaining is {mcmc_iters - mcmc_trials}')
            
            for i in range(self.num_keep+1, self.N):
                for _ in range(mcmc_iters - self.mcmc_trials):
                    dist = MultivariateNormal(part_vals[i, :], cov_matrix)
                    part_vals_prop = dist.sample()

                    prior_curr = self._prior_pdf(part_vals[i, :])
                    prior_prop = self._prior_pdf(part_vals_prop)

                    if torch.isnan(prior_prop / prior_curr) or torch.rand(1) > prior_prop / prior_curr:
                        continue

                    prop = self._trans_finv(part_vals_prop)
                    part_sim_prop = self.simulator(prop)
                    dist_prop = self._dist_func(part_sim_prop)
                    sims_mcmc[i - self.num_keep] += 1

                    if dist_prop <= dist_next:
                        part_vals[i, :] = part_vals_prop
                        part_s[0,i] = dist_prop
                        part_sim[i, :] = part_sim_prop
                        i_acc[i - self.num_keep] += 1
                        
            num_mcmc_iters = max(0, mcmc_iters - self.mcmc_trials) + self.mcmc_trials
            p_acc = i_acc.sum().item() / (num_mcmc_iters * (self.N - self.num_keep))

            print(f'MCMC acceptance probability was {p_acc}')
            
            # Update the total simulations
            sims += sims_mcmc.sum().item()

            # Calculate the new number of MCMC trials
            self.mcmc_trials = (mcmc_iters // 2) + (mcmc_iters % 2 > 0)

            # Compute number of unique particles
            unique_particles = torch.unique(part_vals[:, 0]).numel()
            print(f'The number of unique particles is {unique_particles}')

            # Compute the next distance and maximum distance
            # Sort the particles
            sorted_indices = torch.argsort(part_s)
            part_s = part_s[0,sorted_indices]
            part_vals = part_vals[sorted_indices]
            part_sim = part_sim[sorted_indices]
        
            # if most of the particles are under the final target then don't drop
            if torch.sum(part_s > dist_final).item() < num_drop:
                num_drop = torch.sum(part_s > dist_final).item()
                self.num_keep = self.N - self.num_drop

            # Information about convergence
            dist_t = dist_next  
            p_acc_t = p_acc     

            dist_max = part_s[0,-1]
            dist_next = part_s[0,self.num_keep - 1]  
            
            part_vals = part_vals.reshape([self.N,self.num_params])
            part_sim = part_sim.reshape([self.N, self.part_obs.shape[0]])
            part_s = part_s.reshape([1,self.N])
            
            print(f'The next distance is {dist_next} and the maximum distance is {dist_max} and the number to drop is {num_drop} ')
            print(f'The number of sims is {sims}')

            if (dist_next < dist_final).all():
                dist_next = dist_final
                
            if p_acc < self.p_acc_min:
                part_vals = part_vals.reshape([self.N,self.num_params])
                part_sim = part_sim.reshape([self.N, self.part_obs.shape[0]])
                part_s = part_s.reshape([1,self.N])
                return part_vals, part_sim, part_s

def run_snpe():
    low_para = -1 * torch.ones(num_dim)
    low = torch.cat([low_para, torch.zeros(1)],0)
    high = torch.ones(num_dim+1)
    prior = utils.BoxUniform(low=low, high=high) # set prior distribution

    pancreatic = sio.loadmat("CancerDatasets.mat")['Pancreatic_data']
    x_0 = torch.from_numpy(pancreatic[0:19,0]).to(torch.float32)

    simulators, prior = prepare_for_sbi(model, prior)

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
        torch.save(posterior, 'snpe_bvcbm_iter{}.pth'.format(i+1))

        posterior_sample = posterior.sample((num_simulations,), x=x_0)
        posterior_samples.append(posterior_sample)

        proposal = posterior.set_default_x(x_0)
        
    sio.savemat("posterior_SNPE_bvcbm.mat",{'posterior':posterior_samples})
        
    return posteriors


def run_psnpe(num_dim):
    low_para = -1 * torch.ones(num_dim)
    low = torch.cat([low_para, torch.zeros(1)],0)
    high = torch.ones(num_dim+1)
    prior = utils.BoxUniform(low=low, high=high) # set prior distribution

    pancreatic = sio.loadmat("CancerDatasets.mat")['Pancreatic_data']
    x_0 = torch.from_numpy(pancreatic[0:19,0]).to(torch.float32)
    
    d_param = 7
    
    c_prior = CustomPrior(theta, high, low, flow_dist)
    simulator, prior = prepare_for_sbi(model, c_prior)

    # you can run smc-abc algorithm to reproduce the approximate posterior samples
    adaptive_SMCABC = adaptive_smc_abc()
    part_vals_new, part_sim, _ = adaptive_SMCABC.run_smc_abc(x_0, simulator, num_dim, low, high)
    
    # or use pre-run dataset for the algorithm
    #part_vals_new = sio.loadmat("results_summ_pilot.mat")['part_vals_smc']#trans_finv(part_vals,low,high)
    #part_sim = sio.loadmat("results_summ_pilot.mat")['part_sim_smc']

    theta = torch.from_numpy(part_vals_new).to(torch.float32)
    x = torch.from_numpy(part_sim).to(torch.float32)

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

    data = flow_dist.sample([10000]).cpu()
    
    inference_smc = SNPE_C(prior=prior, density_estimator='nsf')

    num_rounds = 2

    posteriors_smc = []
    posterior_samples_smc = []
    proposal_smc = prior

    num_simulations = 10000

    for i in range(num_rounds):
        if i == 0:
            theta = prior.sample([num_simulations])
            x = simulator(theta).cpu()
            #theta = torch.from_numpy(part_vals_new).to(torch.float32)
            #x = torch.from_numpy(part_sim).to(torch.float32)
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
        torch.save(posteriors_smc, 'psnpe_bvcbm_iter{}.pth'.format(i+1))

        posterior_samples_smc.append(posterior_sample_smc)
        posteriors_smc.append(posterior_smc)
        proposal_smc = posterior_smc.set_default_x(x_0)
        
    sio.savemat("posterior_SNPE_bvcbm.mat",{'posterior':posterior_sample_smc})
    
    return posteriors_smc

if __name__ == '__main__':
    num_dim = 6
    
    run_snpe(num_dim)
    
    run_psnpe(num_dim)