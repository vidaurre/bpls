function [predictedY,stats,predictedYmean] = cvpls(Xin,Yin,parameters,varargin)
% Cross-validation using PLS with possibility of permutation testing
%
% INPUTS
% Xin - input data matrix  (samples X features1)
% Yin - output data matrix (samples X features2)
% parameters is a structure with:
%   + The parameters for PLS (see plsinit.m)
%   + K, a vector with the number of PLS components to evaluate (see plsinit.m)
%   + Nfeatures - Proportion of features from Xin to be initially filtered  
%   + CVscheme - vector of two elements: first is number of folds for model evaluation;
%             second is number of folds for the model selection phase (0 in both for LOO)
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

[N,q] = size(Yin); p = size(Xin,2);
if nargin<3, parameters = {}; end
if ~isfield(parameters,'K'), K = 2:q-1;
else K = parameters.K; end
if ~isfield(parameters,'cyc'), cyc = 0;
else cyc = parameters.cyc; end
if ~isfield(parameters,'initialisation'), initialisation = 'cca';
else initialisation = parameters.initialisation; end
if ~isfield(parameters,'pcaX'), pcaX = 0.99;
else pcaX = parameters.pcaX; end
if ~isfield(parameters,'pcaY'), pcaY = 0.99;
else pcaY = parameters.pcaY; end
if ~isfield(parameters,'CVscheme'), CVscheme=[10 10];
else CVscheme = parameters.CVscheme; end
if ~isfield(parameters,'Nfeatures'), Nfeatures=0;
else Nfeatures = parameters.Nfeatures; end
if ~isfield(parameters,'Nperm'), Nperm=1;
else Nperm = parameters.Nperm; end
if ~isfield(parameters,'riemann'), riemann = length(size(Xin))==3;
else riemann = parameters.riemann; end
if ~isfield(parameters,'keepvar'), keepvar = 1;
else keepvar = parameters.keepvar; end
if ~isfield(parameters,'verbose'), verbose=0;
else verbose = parameters.verbose; end

% if riemann==1 && pcaX>0
%     error('You cannot do PCA on X if riemann is set to 1 - set options.pcaX = 0')
% end

% PLS options
options = struct();
options.cyc = cyc;
options.tol = 0.001;
options.adaptive = 0;
options.initialisation = initialisation;
options.pcaX = 0; options.pcaY = 0;

% put Xin in the right format, which depends on riemann=1
Xin = reshape_pred(Xin,riemann,keepvar); 

% Putting Xin it in tangent space if riemann=1
if riemann
    Xin = permute(Xin,[2 3 1]);
    for j=1:size(Xin,3)
        ev = eig(Xin(:,:,j));
        if any(ev<0)
            error(sprintf('The matrix for subject %d is not positive definite',j))
        end
    end
    Cin = mean_covariances(Xin,'riemann');
    Xin = Tangent_space(Xin,Cin)'; 
else
    Cin = [];
end

% get confounds, and deconfound Xin
confounds=[]; mx1 = zeros(1,p);
if (nargin>5)
    mx1 = mean(Xin);
    Xin = Xin - repmat(mx1,N,1);
    confounds = varargin{3};
    confounds = confounds - repmat(mean(confounds),N,1);
    confounds = confounds ./ repmat(std(confounds),N,1);
    betaX = pinv(confounds)*Xin;  
    Xin = Xin - confounds*betaX;
end

% PCA-ing Xin
mx2 = zeros(1,p);
if pcaX>0
    mx2 = mean(Xin);
    Xin = Xin - repmat(mx2,N,1);
    [A_X,Xin,r_X] = pca(Xin);
    r_X = (cumsum(r_X)/sum(r_X));
    Xin = Xin(:,r_X<pcaX);
    A_X = A_X(:,r_X<pcaX);
    if size(Xin,2) <= any(K)
        error('Some K is higher than the number of principal components for X - decrease pcaX')
    end
end

% Standardizing Xin
mx = mean(Xin);  sx = std(Xin);
Xin = Xin - repmat(mx,N,1);
Xin(:,sx>0) = Xin(:,sx>0) ./ repmat(sx(sx>0),N,1);

if (Nperm<2),  Nperm=1;  end;
cs=[];
if (nargin>3)
    cs=varargin{1};
    if ~isempty(cs)
        if size(cs,2)>1 % matrix format
            [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));    
            [grotMZi(:,2),grotMZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==1));
            [grotDZi(:,2),grotDZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==2));
        else
            allcs = [];
            nz = cs>0; 
            gr = unique(cs(nz));  
            for g=gr'
               ss = find(cs==g);
               for s1=ss,
                   for s2=ss
                       allcs = [allcs; [s1 s2]];
                   end
               end
            end
            % grotMZi and grotDZi need to be computer here
        end
    end
end
if ~exist('allcs','var'), allcs = []; end
PrePerms=0;
if (nargin>4)
    Permutations=varargin{2};
    if ~isempty(Permutations)
        PrePerms=1;
        Nperm=size(Permutations,2);
    end
end

YinORIG = Yin; 
YinORIGmean = zeros(size(Yin));
grotperms = zeros(Nperm,1);
YC = zeros(size(Yin));
YCmean = zeros(size(Yin));

C = inv(cov(Yin) + 1e-5*eye(q));

for perm=1:Nperm
    if (perm>1)
        if isempty(cs)           % simple full permutation with no correlation structure
            rperm = randperm(N);
            Yin=YinORIG(rperm,:);
        elseif (PrePerms==0)          % complex permutation, taking into account correlation structure
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
        else                   % pre-supplied permutation
            Yin=YinORIG(Permutations(:,perm),:);  % or maybe it should be the other way round.....?
        end
    end
    
    predictedYp = zeros(N,q);
    if ~isempty(confounds), predictedYpC = zeros(N,1); end
    
    % create the inner CV structure - stratified for family=multinomial
    folds = cvfolds(Yin,CVscheme(1),allcs);
    if perm==1, chosenK = zeros(length(folds),1); end
    
    for ifold = 1:length(folds) 
        
        if verbose, fprintf('CV iteration %d \n',ifold); end
        J = folds{ifold};
        if isempty(J), continue; end
        
        ji = setdiff(1:N,J); QN = length(ji);
        X = Xin(ji,:); Y = Yin(ji,:);  
        
        % computing mean of the response in the original space
        YinORIGmean(J,:) = repmat(mean(Y),length(J),1);
        
        % precision matrix 
        Cji = inv(cov(Y) + 1e-5*eye(q));
        
        % deconfounding business
        my1 = zeros(1,q);
        if ~isempty(confounds),
            my1 = mean(Y);
            Y = Y - repmat(my1,length(ji),1);
            betaY = pinv(confounds(ji,:))*Y;
            Y = Y - confounds(ji,:)*betaY;
        end
        
        YCmean(J,:) = repmat(mean(Y),length(J),1);
        
        % PCA-ing Y
        my2 = zeros(1,q);
        if pcaY > 0
            my2 = mean(Y); 
            Y = Y - repmat(my2,size(Y,1),1);
            [A_Y,Y,r_Y] = pca(Y);
            r_Y = (cumsum(r_Y)/sum(r_Y));
            Y = Y(:,r_Y<pcaY);
            A_Y = A_Y(:,r_Y<pcaY);
            if size(Y,2) <= any(K)
                error('Some K is higher than the number of principal components for Y - decrease pcaY')
            end
        end
   
        % centering response
        my = mean(Y); Y = Y - repmat(my,size(Y,1),1);

        % pre-kill features
        if Nfeatures<p && Nfeatures>0,
            dev = sum(abs(corr(X,Y)),2);
            [~,groti]=sort(dev);
            groti=groti(end-Nfeatures+1:end);
        else
            groti = find(sx>0);
        end
        
        QXin = Xin(ji,groti); 
        QYin = Yin(ji,:);
        if ~isempty(confounds), Qconfounds=confounds(ji,:); end
        X = X(:,groti);
        
        % family structure for this fold
        Qallcs=[]; 
        if (~isempty(cs)),
            [Qallcs(:,2),Qallcs(:,1)]=ind2sub([length(cs(ji,ji)) length(cs(ji,ji))],find(cs(ji,ji)>0));  
        end
        
        % create the inner CV structure - stratified for family=multinomial
        Qfolds = cvfolds(Y,CVscheme(2),Qallcs);
        
        % parameter selection loop
        Dev = Inf(1,length(K));
        for ik = 1:length(K)
            QpredictedYp = Inf(QN,q);
            options.k = K(ik);
            
            % Inner CV loop to estimate accuracy for k
            for Qifold = 1:length(Qfolds)
                QJ = Qfolds{Qifold};
                Qji = setdiff(1:QN,QJ); 
                QX = QXin(Qji,:); QY = QYin(Qji,:);
                         
                % deconfounding business
                if ~isempty(confounds),
                    Qmy1 = mean(QY);
                    QY = QY - repmat(Qmy1,length(Qji),1);
                    QbetaY = pinv(Qconfounds(ji,:))*QY;
                    QY = QY - Qconfounds(Qji,:)*QbetaY;
                end
                
                % PCA-ing Y
                if pcaY > 0
                    Qmy2 = mean(QY);
                    QY = QY - repmat(Qmy2,length(Qji),1);
                    [A_QY,QY,r_QY] = pca(QY);
                    r_QY = (cumsum(r_QY)/sum(r_QY));
                    QY = QY(:,r_QY<pcaY);
                    A_QY = A_QY(:,r_QY<pcaY);
                    if size(QY,2) <= any(K)
                        error('Some K is higher than the number of principal components for Y - decrease pcaY')
                    end
                end
                
                % centering response
                Qmy = mean(QY); QY = QY - repmat(Qmy,size(QY,1),1);
                
                % train model
                plsfit = plsinit(QX,QY,options);
                if options.cyc>0, plsfit = plsvbinference(QX,QY,plsfit,0); end 
                
                % predict
                QXJ = QXin(QJ,:);
                YDist = plspredict(QXJ,plsfit);
                QpredictedYpQJ = YDist.Mu; + repmat(Qmy,length(QJ),1);
                
                % undo whatever we did before
                if pcaY > 0
                    QpredictedYp(QJ,:) = QpredictedYpQJ * A_QY' + repmat(Qmy2,length(QJ),1);
                else 
                    QpredictedYp(QJ,:) = QpredictedYpQJ;
                end
                if ~isempty(confounds),
                    QpredictedYp(QJ,:) = QpredictedYp(QJ,:) + ...
                        Qconfounds(QJ,:)*QbetaY + repmat(Qmy1,length(QJ),1);
                end
            end
            
            % Pick the one with the lowest deviance in original space 
            d = QpredictedYp - QYin;
            Cd = Cji * d';
            Qdev = zeros(QN,1);
            for n=1:size(QpredictedYp,2)
                Qdev = Qdev + d(:,n).*Cd(n,:)';
            end
            Dev(ik) = mean(Qdev);
        end
        
        [~,I] = min(Dev);
        options.k = K(I);
        if perm==1, chosenK(ifold) = K(I); end

        % train model
        plsfit = plsinit(X,Y,options);
        if options.cyc>0, plsfit = plsvbinference(X,Y,plsfit,0); end

        % predict the test fold
        XJ = Xin(J,groti);
        distY = plspredict(XJ,plsfit);
        predictedYpJ = distY.Mu + repmat(my,length(J),1);
        
        % undo whatever we did before
        if pcaY > 0
            predictedYp(J,:) = predictedYpJ * A_Y' + repmat(my2,length(J),1);
        else
            predictedYp(J,:) = predictedYpJ;
        end
        
        predictedYpC(J,:) = predictedYp(J,:); YC(J,:) = Yin(J,:); % predictedYpC and YC in deconfounded space
        if ~isempty(confounds),
            YC(J,:) = Yin(J,:) - confounds(J,:)*betaY; % deconfound YC
            predictedYp(J,:) = predictedYp(J,:) + confounds(J,:)*betaY + repmat(my1,length(J),1); % confound
        end
                
    end
    
    % grotperms computed in original space
    d = predictedYp - Yin;
    Cd = C * d';
    dev = zeros(N,1);
    for n=1:size(predictedYp,2)
        dev = dev + d(:,n).*Cd(n,:)';
    end    
    grotperms(perm) = mean(dev);
    
    if perm==1
        predictedY = predictedYp;
        predictedYmean = YinORIGmean; 
        stats = {};
        stats.K = chosenK;
        stats.dev = sum((YinORIG-predictedYp).^2);
        stats.nulldev = sum((YinORIG-YinORIGmean).^2);
        if Nperm==1
            [~,pv] = corrcoef(YinORIG,predictedYp); stats.pval=pv(1,2);
        end
        if ~isempty(confounds),
            stats.dev_deconf = sum((YC-predictedYpC).^2);
            stats.nulldev_deconf = sum((YC-YCmean).^2);
            if Nperm==1
                [~,pvd] = corrcoef(YC,predictedYpC); stats.pval_deconf=pvd(1,2);
            end
        end
        stats.cod = 1 - stats.dev / stats.nulldev;
        if ~isempty(confounds), stats.cod_deconf = 1 - stats.dev_deconf / stats.nulldev_deconf; end
    else
        fprintf('Permutation %d \n',perm)
    end
end

if Nperm>1 
    stats.pval = sum(grotperms<=grotperms(1)) / (Nperm+1);
end

end





function folds = cvfolds(Y,CVscheme,allcs)

if nargin<4, allcs = []; end

[N,q] = size(Y);

if CVscheme==0, nfolds = N;
else nfolds = CVscheme;
end
folds = {}; ifold = 1;
grotDONE = zeros(N,1);

for k = 1:nfolds
    if sum(grotDONE)==N, break; end
    j=1;  folds{ifold} = [];
    while length(folds{ifold}) < ceil(N/nfolds) && j<=N
        if (grotDONE(j)==0)
            folds{ifold}=[folds{ifold} j];
            if (~isempty(allcs))  % leave out all samples related to the one in question
                if size(find(allcs(:,1)==j),1)>0
                    folds{ifold}=[folds{ifold} allcs(allcs(:,1)==j,2)'];
                end
            end
            grotDONE(folds{ifold})=1;
        end
        j=j+1;
        if k>1 && k<nfolds,
            if sum(grotDONE)>k*N/nfolds
                break
            end
        end
    end
    if ~isempty(folds{ifold}), ifold = ifold + 1; end
end

end


