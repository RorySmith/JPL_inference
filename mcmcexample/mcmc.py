import numpy as np

import emcee


import pylab as plt


## signal model (sine wave with amplitude A and frequency f)
def h(A,t,f):
    
    omega = np.pi*2.*f
    return A*np.sin(omega*t)

## generate some data: data = signal + Gaussian noise

## signal amplitude:
A0 = 0.3
## signal frequency (Hz)
f0 = 10 
srate = 1./1024. #sample rate of the data (Hz)
times = np.arange(1,10,srate)

signal = h(A0,times,f0)
noise = np.random.normal(loc=0,scale=1,size=len(signal))
data = signal + noise

### define log prior, likelihood and posterior probability 

def log_prior(theta):
    A,f = theta
    if A > 5. or A < 0. or f > 15. or f < 5.:
	return -np.inf
    else:

     	return 1  # flat prior

def log_likelihood(theta, data):
    A,f = theta
    return -0.5 * np.sum( (data - h(A,times,f)) ** 2 / 1. ** 2) # prob of noise (data - candidate signal) : (unnormalized) normal dist with mean = 0, sigma = 1

def log_posterior(theta, data):
    return log_prior(theta) + log_likelihood(theta, data) 


ndim = 2  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 5000  # "burn-in" period to let chains stabilize
nsteps = nburn*2.  # number of MCMC steps to take

# we'll start at random locations between 0 and 10 
starting_guesses = 10 * np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data])
sampler.run_mcmc(starting_guesses, nsteps)

import corner
samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
print samples
fig = corner.corner(samples, labels=['A','f'],
                      truths=[A0, f0])
fig.savefig("triangle.png")




