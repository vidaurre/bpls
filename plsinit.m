function [model] = plsinit(X,Y,options)
% function [model] = plsinit (X,Y,model)
%
% Initialise observation model
%
% X             Inputs
% Y             Outputs
% options       Training options
% - k: number of latent components.
% - pcaX: if specified higher than 0, PLS will use a lowrank version of X; 
%       then, pcaX (between 0 and 1) indicates the proportion of variance explained from X  
% - pcaY: if specified higher than 0, PLS will use a lowrank version of Y; 
%       then, pcaY (between 0 and 1) indicates the proportion of variance explained from Y
% - adaptive: should adaptiveness be use?
% - initialisation: strategy to init the latent components: 
%       'cca', 'pca', 'pls' or 'random', which respectively use the matlab routines 
%       canoncorr, pca, plsregress and randn.
% - tol: threshold in the decrement of the free energy to stop the variational loop.
% - cyc: maximum number of variational iterations.
%
% Author: Diego Vidaurre, University of Oxford

model = paramchk(options);
if (model.options.pcaX > 0 && model.options.pcaX < 1) || ... 
        (model.options.pcaY > 0 && model.options.pcaY < 1)  
    [X,Y,model.pca] = pcaXY(X,Y,model.options);
end
model=initpriors(X,Y,model,options);
model=initpost(X,Y,model);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [model] = initpriors(X,Y,model,options)

p = size(X,2);
q = size(Y,2);
k = model.options.k;

if isfield(options,'prior')
    model.prior = options.prior;
end
model.prior=struct();
model.prior.sigma = struct('Gam_shape',[],'Gam_rate',[]);
model.prior.sigma.Gam_shape = 0.01;
model.prior.sigma.Gam_rate = 0.01 * ones(p,1);
model.prior.gamma = struct('Gam_shape',[],'Gam_rate',[]);
model.prior.gamma.Gam_shape = 0.01;
model.prior.gamma.Gam_rate = 0.01 * ones(k,1);
model.prior.Omega = struct('Gam_shape',[],'Gam_rate',[]);
model.prior.Omega.Gam_shape = k - 0.5;
model.prior.Omega.Gam_rate = (k - 0.5) * ones(1,k);
model.prior.Psi = struct('Gam_shape',[],'Gam_rate',[]);
model.prior.Psi.Gam_shape = q - 0.5;
model.prior.Psi.Gam_rate = (q - 0.5) * ones(1,q) ;
if model.options.adaptive
    model.prior.phi = struct('Gam_shape',[],'Gam_rate',[]);
    model.prior.phi.Gam_shape = 0.01;
    model.prior.phi.Gam_rate = 0.01;
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [model] = initpost(X,Y,model)

k = model.options.k;
[N,p]=size(X);
q = size(Y,2);

if strcmp(model.options.initialisation,'pls')
    [model.P.Mu_P,model.Q.Mu_Q,model.Z.Mu_Z] = plsregress(X,Y,k);
    model.Q.Mu_Q = model.Q.Mu_Q';
elseif strcmp(model.options.initialisation,'pca')
    [model.P.Mu_P, model.Z.Mu_Z] = pca(X,'NumComponents', k );
    %model.Q.Mu_Q = (model.Z.Mu_Z' * model.Z.Mu_Z + 0.1 * eye(k)) \ model.Z.Mu_Z' * Y;
    model.Q.Mu_Q = model.Z.Mu_Z \ Y;
elseif strcmp(model.options.initialisation,'pcay')
    [model.Q.Mu_Q, model.Z.Mu_Z] = pca(Y,'NumComponents', k );
    model.Q.Mu_Q = model.Q.Mu_Q';
    model.P.Mu_P = (X' * X + 0.1 * eye(p)) \ X' * model.Z.Mu_Z;
elseif strcmp(model.options.initialisation,'cca')
    [model.P.Mu_P,~,~,model.Z.Mu_Z] = canoncorr(X,Y);
    model.P.Mu_P = model.P.Mu_P(:,1:k);
    model.Z.Mu_Z = model.Z.Mu_Z(:,1:k);
    %model.Q.Mu_Q = (model.Z.Mu_Z' * model.Z.Mu_Z + 0.1 * eye(k)) \ model.Z.Mu_Z' * Y;
    model.Q.Mu_Q = model.Z.Mu_Z \ Y;
elseif strcmp(model.options.initialisation,'random')
    model.Z.Mu_Z = randn(N, k );
    model.P.Mu_P = (X' * X + 0.1 * eye(p)) \ X' * model.Z.Mu_Z;
    model.Q.Mu_Q = (model.Z.Mu_Z' * model.Z.Mu_Z + 0.1 * eye(k)) \ model.Z.Mu_Z' * Y;
else
    error('Unknown Initialisation \n')
end
model.Z.S_Z = eye(k); %inv(cov(model.Z.Mu_Z));
model.P.S_P = zeros(k,p,p);
model.Q.S_Q = zeros(q,k,k);
for l=1:k, model.P.S_P(l,:,:) = eye(p); end
for j=1:q, model.Q.S_Q(j,:,:) = eye(k); end

%model.Omega.Gam_rate = model.prior.Omega.Gam_rate +  0.5 * sum( (model.Z.Mu_Z - X * model.P.Mu_P).^2 );
%model.Omega.Gam_shape = model.prior.Omega.Gam_shape + N / 2;
model.Omega.Gam_shape = 1;
model.Omega.Gam_rate = ones(1,k);

model.Psi.Gam_rate = model.prior.Psi.Gam_rate +  0.5 * sum( (Y - model.Z.Mu_Z * model.Q.Mu_Q).^2 );
model.Psi.Gam_shape = model.prior.Psi.Gam_shape + N / 2;
model.sigma.Gam_shape = model.prior.sigma.Gam_shape + k/2;
model.sigma.Gam_rate = model.prior.sigma.Gam_rate + 0.5 * sum(model.P.Mu_P.^2, 2)  ;
if model.options.adaptive
    model.phi.Gam_shape = model.prior.phi.Gam_shape + (p*k)/2;
    model.phi.Gam_rate = model.prior.phi.Gam_rate +0.5 * sum(sum(model.Q.Mu_Q.^2));
    model.gamma.Gam_shape = model.prior.gamma.Gam_shape;
    model.gamma.Gam_rate = model.prior.gamma.Gam_rate + zeros(k,1);
else
    model.gamma.Gam_shape = model.prior.gamma.Gam_shape + q/2;
    model.gamma.Gam_rate = model.prior.gamma.Gam_rate + 0.5 * sum(model.Q.Mu_Q.^2, 2);
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [model]=paramchk(options)

model = struct();
if isfield(options,'k')
    model.options.k = options.k;
else
    error('Error in hmmtrain: k not specified');
end
if ~isfield(options,'cyc')
    model.options.cyc = 100;
else
    model.options.cyc = options.cyc;
end
if ~isfield(options,'tol')
    model.options.tol = 0.001;
else
    model.options.tol = options.tol;
end
if ~isfield(options,'adaptive')
    model.options.adaptive = 0;
else
    model.options.adaptive = options.adaptive;
end
if ~isfield(options,'initialisation')
    model.options.initialisation = 'cca';
else
    model.options.initialisation = options.initialisation;
end
if ~isfield(options,'pcaX')
    model.options.pcaX = 0;
else
    model.options.pcaX = options.pcaX;
end
if ~isfield(options,'pcaY')
    model.options.pcaY = 0;
else
    model.options.pcaY = options.pcaY;
end
% if ~isfield(options,'deflating')
%     model.options.deflating = 0;
% end

end

