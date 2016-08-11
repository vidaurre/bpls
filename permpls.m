function [pval,pval_q,R2,R2p,Yhat] = permpls(Xin,Yin,parameters,varargin)
% Permutation testing over PLS
%
% INPUTS
% Xin - input data matrix  (samples X features1)
% Yin - output data matrix (samples X features2)
% parameters is a structure with:
%   + The parameters for PLS (see plsinit.m)
%   + Nperm - number of permutations (set to 0 to skip permutation testing)
%   + verbose -  display progress?
% correlation_structure (optional) - A (Nsamples X Nsamples) matrix with
%                                   integer dependency labels (e.g., family structure), or 
%                                    A (Nsamples X 1) vector defining some
%                                    grouping: (1...no.groups) or 0 for no group
% Permutations (optional but must also have correlation_structure) - pre-created set of permutations
% confounds (optional) - features that potentially influence the inputs, and the outputs for family="gaussian'
%
% OUTPUTS
% + pval -  p-value resulting of performing permutation testing on PCA+CCA
% + R2 - explained variance for each variable in Yin
% + R2p - explained variance for each variable in Yin and permutation
%
% Diego Vidaurre, University of Oxford (2016)

if ~isfield(parameters,'Nperm'), Nperm=1000;
else Nperm = parameters.Nperm; end
if ~isfield(parameters,'standardiseX'), standardiseX=1;
else standardiseX = parameters.standardiseX; end
if ~isfield(parameters,'standardiseY'), standardiseY=1;
else standardiseY = parameters.standardiseY; end
if ~isfield(parameters,'cyc'), cyc=0;
else cyc = parameters.cyc; end
if ~isfield(parameters,'verbose'), verbose = 0;
else verbose = parameters.verbose; end

% keep only NaN-free subjects
keep = (~isnan(sum(Xin,2))) & (~isnan(sum(Yin,2)));
Xin = Xin(keep,:); Yin = Yin(keep,:); 
[N,p] = size(Xin); q = size(Yin,2);

% If X or Y are probabilities, remove one column to make them full rank
% if all( abs(sum(Xin,2)-1)<1e-8  )
%     Xin = Xin(:,1:end-1); p = p - 1;
% end
% if all( abs(sum(Yin,2)-1)<1e-8  )
%     Yin = Yin(:,1:end-1); q = q - 1; 
% end
X = Xin; Y = Yin;

% Standardize 
if standardiseX
    Xin = Xin - repmat(mean(Xin),N,1);
    Xin = Xin ./ repmat(std(Xin),N,1);
end
if standardiseY
    Yin = Yin - repmat(mean(Yin),N,1);
    Yin = Yin ./ repmat(std(Yin),N,1);
end
    
% Family structure
cs=[];
if (nargin>3)
    cs=varargin{1};
    if ~isempty(cs) %assumed to be in matrix format
        cs = cs(keep,keep);
        [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));
        [grotMZi(:,2),grotMZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==1));
        [grotDZi(:,2),grotDZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==2));
    end
end
if ~exist('allcs','var'), allcs = []; end

% Pre-existing permutations?
PrePerms=0;
if (nargin>4)
    Permutations=varargin{2};
    if ~isempty(Permutations)
        PrePerms=1;
        Nperm=size(Permutations,2);
    end
end

% Confounds
if (nargin>5)
    confounds = varargin{3};
    confounds = confounds - repmat(mean(confounds),N,1);
    [~,Xin,~,Yin] = nets_deconfound(Xin,Yin,confounds,'gaussian',[],[]);
    % Standardize again
    if standardiseX
        Xin = Xin - repmat(mean(Xin),N,1);
        Xin = Xin ./ repmat(std(Xin),N,1);
    end
    if standardiseY
        Yin = Yin - repmat(mean(Yin),N,1);
        Yin = Yin ./ repmat(std(Yin),N,1);
    end
end

% PLS for original data
R2p = zeros(Nperm,q);
R2 = zeros(1,q);
fit = plsinit(Xin,Yin,parameters);
if cyc>0
    fit = plsvbinference(Xin,Yin,fit,0);
end
Yhat = plspredict (Xin,fit);

my = mean(Yin);
R2 = 1 - sum((Yin - Yhat.Mu).^2) ./ sum((Yin - repmat(my,N,1)).^2);


% Permute
YinORIG=Yin; 
for perm=1:Nperm
    if PrePerms==1  % pre-supplied permutation
        Yin=YinORIG(Permutations(:,perm),:);  % or maybe it should be the other way round.....?
    elseif isempty(cs)           % simple full permutation with no correlation structure
        rperm = randperm(N);
        Yin=YinORIG(rperm,:);
    else         % complex permutation, with correlation structure
        PERM=zeros(1,N);
        perm1=randperm(size(grotMZi,1));
        for ipe=1:length(perm1)
            if rand<0.5, wt=[1 2]; else wt=[2 1]; end;
            PERM(grotMZi(ipe,1))=grotMZi(perm1(ipe),wt(1));
            PERM(grotMZi(ipe,2))=grotMZi(perm1(ipe),wt(2));
        end
        perm1=randperm(size(grotDZi,1));
        for ipe=1:length(perm1)
            if rand<0.5, wt=[1 2]; else wt=[2 1]; end;
            PERM(grotDZi(ipe,1))=grotDZi(perm1(ipe),wt(1));
            PERM(grotDZi(ipe,2))=grotDZi(perm1(ipe),wt(2));
        end
        from=find(PERM==0);  pto=randperm(length(from));  to=from(pto);  PERM(from)=to;
        Yin=YinORIG(PERM,:);
    end
    if mod(perm,10)==0 && verbose, fprintf('iteration %d \n',perm); end
    
    fit = plsinit(Xin,Yin,parameters);
    if cyc>0
        fit = plsvbinference(Xin,Yin,fit,0);
    end
    Yhatp = plspredict (Xin,fit);
    R2p(perm,:) = 1 - sum((Yin - Yhatp.Mu).^2) ./ sum((Yin - repmat(my,N,1)).^2);
end

R2sum = sum(R2); R2psum = sum(R2p,2);

pval = ( sum(R2psum>=R2sum) + 1 ) / (Nperm+1);
pval_q = zeros(1,q);
for j=1:q
    pval_q(j) = ( sum(R2p(:,j)>=R2(j)) + 1 ) / (Nperm+1);
end

end