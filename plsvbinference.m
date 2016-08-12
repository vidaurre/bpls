function [model,fe] = plsvbinference (X,Y,model,verbose,XX)
% function [model] = plstrain (X,Y,model)
%
% Update observation model
%
% X             Inputs
% Y             Outputs
% Model         An initialised model
% trace         want to report each iteration?
% 
% Output
% model         the fitted Bayesian PLS model
% fe            free energy 
%
% Author: Diego Vidaurre, University of Oxford

if isfield(model.pca,'A_X'), 
    X = X - repmat(model.pca.mx,size(X,1),1);
    X = X * model.pca.A_X; 
end
if isfield(model.pca,'A_Y'), 
    Y = Y - repmat(model.pca.my,size(Y,1),1);
    Y = Y * model.pca.A_Y; 
end

[N,p]=size(X);
q = size(Y,2);

if nargin<3, model = plsinit(X,Y); end
if nargin<4, verbose = 1; end
if nargin<5, XX = X' * X; end 

k = model.options.k;

it = 1;
fe = Inf; 

model.gamma.Gam_shape = 1;
model.gamma.Gam_rate = 100000*ones(k,1);
model.sigma.Gam_shape = 1;
model.sigma.Gam_rate = 100000*ones(p,1);
model.Psi.Gam_shape = 1;
model.Psi.Gam_rate = ones(1,q);

for it=1:model.options.cyc
    
    model.phi.Gam_rate = 1; model.phi.Gam_shape =1;
    
    fe_old = fe;
     
    % Z
    QPsi = model.Q.Mu_Q * diag(model.Psi.Gam_shape ./ model.Psi.Gam_rate);
    QPsiQ = QPsi * model.Q.Mu_Q';
    for j=1:q,
        QPsiQ = QPsiQ + (model.Psi.Gam_shape / model.Psi.Gam_rate(j)) * permute(model.Q.S_Q(j,:,:),[2 3 1]);
    end 
    OmegaP = diag(model.Omega.Gam_shape ./ model.Omega.Gam_rate) * model.P.Mu_P';
    model.Z.S_Z = inv(diag(model.Omega.Gam_shape ./ model.Omega.Gam_rate) + QPsiQ);
    model.Z.Mu_Z = (model.Z.S_Z * ( OmegaP * X' + QPsi * Y' ))';
    
    % P
    for l=1:k
        if model.options.adaptive
            S = inv(diag( model.sigma.Gam_shape ./  model.sigma.Gam_rate ) .* ...
                ( (model.gamma.Gam_shape /  model.gamma.Gam_rate(l)) * eye(p) ) + ...
                (model.Omega.Gam_shape / model.Omega.Gam_rate(l)) * XX);
        else
            S = inv(diag( model.sigma.Gam_shape ./  model.sigma.Gam_rate ) + ...
                (model.Omega.Gam_shape / model.Omega.Gam_rate(l)) * XX);
        end
        model.P.S_P(l,:,:) = S;
        model.P.Mu_P(:,l) = S * (model.Omega.Gam_shape / model.Omega.Gam_rate(l)) * X' * model.Z.Mu_Z(:,l);
    end;
    
    % Omega
    if 0
        e = (model.Z.Mu_Z - X * model.P.Mu_P).^2;
        swx2 = zeros(N,k);
        for l=1:k
            swx2(:,l) = sum(( X * permute(model.P.S_P(l,:,:),[2 3 1]) ) .* X,2);
        end;
        model.Omega.Gam_rate = model.prior.Omega.Gam_rate + ...
            0.5 * sum(e + swx2 + repmat(diag(model.Z.S_Z)', N, 1)) ;
        model.Omega.Gam_shape = model.prior.Omega.Gam_shape + N / 2;
    end
    
    % sigma
    model.sigma.Gam_shape = model.prior.sigma.Gam_shape + k/2;
    model.sigma.Gam_rate = model.prior.sigma.Gam_rate + 0.5 * sum(model.P.Mu_P.^2, 2);
    for l=1:k
        if model.options.adaptive
            model.sigma.Gam_rate = model.sigma.Gam_rate + 0.5 * ...
                (model.gamma.Gam_shape / model.gamma.Gam_rate(l)) * ...
                (model.P.Mu_P(:,l).^2 +  diag(permute(model.P.S_P(l,:,:),[2 3 1])));
        else
            model.sigma.Gam_rate = model.sigma.Gam_rate + 0.5 * diag(permute(model.P.S_P(l,:,:),[2 3 1]));
        end
    end
        
    % Q
    SZZ = N * model.Z.S_Z;
    EZZ = model.Z.Mu_Z' * model.Z.Mu_Z + SZZ;
    for j=1:q,
        model.Q.S_Q(j,:,:) = inv( (model.phi.Gam_shape /  model.phi.Gam_rate) * ...
            diag(model.gamma.Gam_shape ./ model.gamma.Gam_rate) ...
                + (model.Psi.Gam_shape / model.Psi.Gam_rate(j)) * EZZ );
        model.Q.Mu_Q(:,j) = (model.Psi.Gam_shape / model.Psi.Gam_rate(j)) * ...
            permute(model.Q.S_Q(j,:,:),[2 3 1]) * model.Z.Mu_Z' * Y(:,j);
    end;
    
    % Psi
    e = sum( (Y - model.Z.Mu_Z * model.Q.Mu_Q).^2 );
    svz2 = zeros(1,q);
    for j=1:q
        svz2(j) = model.Q.Mu_Q(:,j)' * SZZ * model.Q.Mu_Q(:,j) + ...
            trace( permute(model.Q.S_Q(j,:,:),[2 3 1]) * EZZ );
    end;
    model.Psi.Gam_shape = model.prior.Psi.Gam_shape + N / 2;
    model.Psi.Gam_rate = model.prior.Psi.Gam_rate + 0.5 * (e + svz2);
    
    %gamma
    if model.options.adaptive
        model.gamma.Gam_shape = model.prior.gamma.Gam_shape + (p+q)/2;
        model.gamma.Gam_rate = model.prior.gamma.Gam_rate + ...
            0.5 * (model.phi.Gam_shape /  model.phi.Gam_rate) * sum(model.Q.Mu_Q.^2, 2) + ...
            0.5 * sum(model.P.Mu_P.^2 .* repmat(model.sigma.Gam_shape ./  model.sigma.Gam_rate,1,k), 1)';
        for j=1:q,
            model.gamma.Gam_rate = model.gamma.Gam_rate + ...
                0.5 * (model.phi.Gam_shape /  model.phi.Gam_rate) * diag(permute(model.Q.S_Q(j,:,:),[2 3 1]));
        end;
        for l=1:k
            model.gamma.Gam_rate(l) = model.gamma.Gam_rate(l) + ...
                0.5 * sum( diag(permute(model.P.S_P(l,:,:),[2 3 1])) .* ...
                (model.sigma.Gam_shape ./  model.sigma.Gam_rate) );
        end;  
    else
        model.gamma.Gam_shape = model.prior.gamma.Gam_shape + q/2;
        model.gamma.Gam_rate = model.prior.gamma.Gam_rate + 0.5 * sum(model.Q.Mu_Q.^2, 2); 
        for j=1:q,
            model.gamma.Gam_rate = model.gamma.Gam_rate + 0.5 * diag(permute(model.Q.S_Q(j,:,:),[2 3 1]));
        end;
    end
    
    % phi
    if model.options.adaptive
        model.phi.Gam_shape = model.prior.phi.Gam_shape + (q*k)/2;
        model.phi.Gam_rate = model.prior.phi.Gam_rate + 0.5 * ...
            sum( (model.gamma.Gam_shape ./ model.gamma.Gam_rate) .* sum(model.Q.Mu_Q.^2, 2) );
        for j=1:q
            model.phi.Gam_rate = model.phi.Gam_rate + 0.5 * ...
                sum( (model.gamma.Gam_shape ./ model.gamma.Gam_rate) .* ...
                diag(permute(model.Q.S_Q(j,:,:),[2 3 1])));
        end;
    end    

    fe = plsfenergy(X,Y,model);
    feIncr = fe_old - fe;
    
    mesgstr='';
    %if feIncr< 0,
    %    mesgstr='(Violation)';
    %end;
    if verbose, fprintf('Iteration %i, Free energy = %f %s \n',it,fe,mesgstr); end
    
    %%% termination conditions
    if abs(feIncr/fe_old*100) <model.options.tol, break; end
   
end;


