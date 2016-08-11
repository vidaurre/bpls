Package bpls
Bayesian Partial Least Squares
Author and maintainer: Diego Vidaurre, University of Oxford <diego.vidaurre at ohba.ox.ac.uk>
Method published in Vidaurre, D., van Gerven M., Bielza, C. & Larrañaga, P & Heskes, T. (2013). Bayesian Sparse Partial Least Squares. Neural Computation.
Published: December 2013.
Matlab version: 8.0.0.783 (R2012b) - It should work for a wide range of matlab versions as far as they implement plsregress and pca.

The current version includes three main functions:

- plsinit.m: initialises the model, and receives the training options.
- plsvbinference.m: trains the model.
- plsfenergy.m: computes de free energy of the model, and is called at each iteration by plsvbinference.m.

See example.m for an example of usage.

The structure 'options' can include the following fields:

- k: number of latent components.
- adaptive: should adaptiveness be use? 
- initialisation: strategy to init the latent components: 'pca', 'pls' or 'random', which respectively use the matlab routines pca, plsregress and randn. 
- tol: threshold in the decrement of the free energy to stop the variational loop.
- cyc: maximum number of variational iterations.

The output structure contains the fields P, Q, Z, sigma, gamma, phi, Omega and Psi, that correspond to the variables defined in the paper, as well as a structure 'options' with a copy of the corresponding structure supplied by the user. It also contains a structure with the priors for each model variable. 

Note: the package implements a different strategy for the adaptive model than that presented in the paper above. The prior used in the paper is not conjugate with the likelihood and inference is only approximate, in the sense that it does not necessarily lead to the solution with the lowest free energy (although works well in practice, it tends to overshrink slightly the regression coefficients). In this version of the code, the elements in matrix P are distributed N(0,\sigma_i \gamma_l), and the elements in the matrix Q are distributed (0,\phi \gamma_l).
