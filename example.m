Repet = 100; % number of repetitions of the experiment
N = 40;  % number of data cases for training
Nt = 1000; % number of data cases for testing

cof0 = zeros(Repet,1);  % coefficient of determination of OLS
cof1 = zeros(Repet,1);  % coefficient of determination for BPLS with random init
cof2 = zeros(Repet,1);  % coefficient of determination for BPLS with PCA init
cof3 = zeros(Repet,1);  % coefficient of determination for BPLS with PLS init
cof4 = zeros(Repet,1);  % coefficient of determination for adaptive BPLS with PCA init 


for r=1:Repet
    
    X = randn(N,20);
    W = randn(20,3); V = randn(3,8);
    Y = X * W * V + randn(N,8);
    Xt = randn(Nt,20); Yt = Xt * W * V + randn(Nt,8);
    
    % OLS
    Beta = zeros(20,8); 
    for i=1:8, Beta(:,i) = pinv(X) * Y(:,i); end
    Yhat0 = Xt * Beta; 
    cof0(r) = mean(1- sum (  (Yhat0 - Yt).^2)  ./ sum( (Yt - repmat(mean(Yt),Nt,1)).^2 )  );
    fprintf('Iteration %i, OLS: %f \n',r,cof0(r));
    
    options.cyc = 5000;
    options.tol = 0.001;
    options.k = 3;
    options.initialisation = 'random';
    options.adaptive = 0;
    
    % BPLS, with random init
    model1 = plsinit(X,Y,options);
    model1 = plsvbinference(X,Y,model1,0);
    yhat1 = plspredict(Xt,model1);
    cof1(r) = mean(1- sum (  (yhat1.Mu - Yt).^2)  ./ sum( (Yt - repmat(mean(Yt),Nt,1)).^2 )  );
    fprintf('Iteration %i, random init: %f \n',r,cof1(r));
    
    % BPLS, with PCA initialisation
    options.initialisation = 'pca';
    model2 = plsinit(X,Y,options);
    model2 = plsvbinference(X,Y,model2,0);
    yhat2 = plspredict(Xt,model2);
    cof2(r) = mean(1- sum (  (yhat2.Mu - Yt).^2)  ./ sum( (Yt - repmat(mean(Yt),Nt,1)).^2 )  );
    fprintf('Iteration %i,  PCA init: %f \n',r,cof2(r));
    
    % BPLS, with PLS initialisation
    options.initialisation = 'pls';
    model3 = plsinit(X,Y,options);
    model3 = plsvbinference(X,Y,model3,0);
    yhat3 = plspredict(Xt,model3);
    cof3(r) = mean(1- sum (  (yhat3.Mu - Yt).^2)  ./ sum( (Yt - repmat(mean(Yt),Nt,1)).^2 )  );
    fprintf('Iteration %i, PLS init: %f \n',r,cof3(r));
    
    % BPLS, with PCA initialisation and adaptive parameter
    options.adaptive = 1;
    options.initialisation = 'pca';
    model4 = plsinit(X,Y,options);
    model4 = plsvbinference(X,Y,model4,0);
    yhat4 = plspredict(Xt,model4);
    cof4(r) = mean(1- sum (  (yhat4.Mu - Yt).^2)  ./ sum( (Yt - repmat(mean(Yt),Nt,1)).^2 )  );
    fprintf('Iteration %i, PCA init and adaptive: %f \n',r,cof4(r));
    
end


