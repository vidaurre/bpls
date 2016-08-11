function [Yhat,YhatPCA] = plspredict (X,model)
% function [model] = plspredict (X,model)
%
% Update observation model
%
% X             Inputs
% Model         A trained model
%
% yhat has fields Mu and S with mean and covariance matrix
%
% Author: Diego Vidaurre, University of Oxford

if isfield(model,'pca') && isfield(model.pca,'A_X'), 
    X = X - repmat(model.pca.mx,size(X,1),1);
    X = X * model.pca.A_X; 
end

Yhat.Mu = X * model.P.Mu_P * model.Q.Mu_Q;
Yhat.S = diag(model.Psi.Gam_shape ./ model.Psi.Gam_rate) + ...
    model.Q.Mu_Q' * diag(model.Omega.Gam_shape ./ model.Omega.Gam_rate) * model.Q.Mu_Q; 

if isfield(model,'pca') && isfield(model.pca,'A_Y'), 
    N = size(X,1);
    YhatPCA = Yhat; 
    Yhat.Mu = Yhat.Mu * model.pca.A_Y' + repmat(model.pca.my,N,1); 
    Yhat.S = model.pca.A_Y * Yhat.S * model.pca.A_Y'; 
end 