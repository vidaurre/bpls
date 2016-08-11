function [FrEn] = plsfenergy (X,Y,model)
% [FrEn] = plsfenergy (X,Y,model)
%
% Computes the Free Energy of an PLS model
%
% X             Inputs
% Y             Outputs
% Model         A PLS model
%
% Author: Diego Vidaurre, University of Oxford

[N,p]=size(X);
q = size(Y,2);
k=model.options.k;


% Neg-Entropy of Z
NegEnt = - 0.5*N*k*(1+log(2*pi)) - 0.5*N*log(det(model.Z.S_Z)); %% all the same
%model.Z.S_Z
%FrEn

% KL divergences
KlDv = 0;
% sigma
for i=1:p
    KlDv = KlDv + gamma_kl(model.sigma.Gam_shape, model.prior.sigma.Gam_shape,...
        model.sigma.Gam_rate(i), model.prior.sigma.Gam_rate(i));
end;
% gamma
for l=1:k
    KlDv = KlDv + gamma_kl(model.gamma.Gam_shape, model.prior.gamma.Gam_shape,...
        model.gamma.Gam_rate(l), model.prior.gamma.Gam_rate(l));
end;
% phi
if model.options.adaptive
    KlDv = KlDv + gamma_kl(model.phi.Gam_shape, model.prior.phi.Gam_shape,...
        model.phi.Gam_rate, model.prior.phi.Gam_rate);
end
% Omega 
%for l=1:k
%    KlDv = KlDv + gamma_kl(model.Omega.Gam_shape,model.prior.Omega.Gam_shape, ...
%        model.Omega.Gam_rate(l),model.prior.Omega.Gam_rate(l));
%end;
% Psi
for j=1:q
    KlDv = KlDv + gamma_kl(model.Psi.Gam_shape,model.prior.Psi.Gam_shape, ...
        model.Psi.Gam_rate(j),model.prior.Psi.Gam_rate(j));
end;
% P
if model.options.adaptive
    for l=1:k
        KlDv = KlDv + gauss_kl(model.P.Mu_P(:,l),zeros(p,1),permute(model.P.S_P(l,:,:),[2 3 1]), ...
            ( (model.gamma.Gam_rate(l) /  model.gamma.Gam_shape) * eye(p)) .* diag(model.sigma.Gam_rate ./ model.sigma.Gam_shape)   );
    end;
else
    for l=1:k
        KlDv = KlDv + gauss_kl(model.P.Mu_P(:,l),zeros(p,1),permute(model.P.S_P(l,:,:),[2 3 1]), ...
            diag(model.sigma.Gam_rate ./ model.sigma.Gam_shape));
    end;
end
% Q
KlDv = KlDv + gauss_kl(model.Q.Mu_Q(:,j), zeros(k,1), permute(model.Q.S_Q(j,:,:),[2 3 1]), ...
    (model.phi.Gam_rate ./  model.phi.Gam_shape) * diag(model.gamma.Gam_rate ./ model.gamma.Gam_shape)   );
%KlDv
% average likelihood

% likelihood for Y
ltpi1= q/2 * log(2*pi);
ldetWishB1=0;
PsiWish_alphasum1=q*0.5*digamma(model.Psi.Gam_shape);
for j=1:q
    ldetWishB1=ldetWishB1+0.5*log(model.Psi.Gam_rate(j));
end;
d = (Y - model.Z.Mu_Z * model.Q.Mu_Q);
Cd = diag(model.Psi.Gam_shape ./ model.Psi.Gam_rate) * d';
dist=zeros(N,1);
for j=1:q,
    dist=dist-0.5*d(:,j).*Cd(j,:)';
end
NormWishtrace=zeros(N,1);
for j=1:q
    NormWishtrace = NormWishtrace + 0.5 * (model.Psi.Gam_shape / model.Psi.Gam_rate(j)) * ...
        ( sum( (model.Z.Mu_Z * permute(model.Q.S_Q(j,:,:),[2 3 1])) .* model.Z.Mu_Z, 2) + ...
        model.Q.Mu_Q(:,j)' * model.Z.S_Z * model.Q.Mu_Q(:,j) + ...
        sum(sum( permute(model.Q.S_Q(j,:,:),[2 3 1]) .* model.Z.S_Z ,2)) );
end;
avLLY = N * (-ltpi1-ldetWishB1+PsiWish_alphasum1) + sum(dist - NormWishtrace);

% likelihood for Z
ltpi2= k/2 * log(2*pi);
ldetWishB2=0;
PsiWish_alphasum2=k*0.5*digamma(model.Omega.Gam_shape);
for l=1:k
    ldetWishB2=ldetWishB2+0.5*log(model.Omega.Gam_rate(l));
end;
d = ( model.Z.Mu_Z - X * model.P.Mu_P);
Cd = diag(model.Omega.Gam_shape ./ model.Omega.Gam_rate) * d';
dist=zeros(N,1);
for l=1:k,
    dist=dist-0.5*d(:,l).*Cd(l,:)';
end
NormWishtrace=zeros(N,1);
for l=1:k,
    NormWishtrace = NormWishtrace + 0.5 * (model.Omega.Gam_shape / model.Omega.Gam_rate(l)) * ...
        (sum( (X * permute(model.P.S_P(l,:,:),[2 3 1])) .* X,2) + model.Z.S_Z(l,l));
end;
avLLZ = N * (-ltpi2-ldetWishB2+PsiWish_alphasum2) + sum(dist - NormWishtrace);


FrEn = NegEnt + KlDv - avLLY - avLLZ;
%fprintf('%f + %f  %f   %f:  %f \n',NegEnt,KlDv,avLLY,avLLZ,FrEn)
