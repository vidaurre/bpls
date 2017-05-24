function [X,Y,s] = pcaXY(X,Y,m)
% It does not scale the variables in X or Y, so if the scales
%  are very unbalanced then a few variables in X/Y can dominate PCA

% if isstruct(X)
%     if isfield(m.pca,'A_X'), 
%         X = X - repmat(m.pca.mx,1,size(X,2));
%         X = X * m.pca.A_X; 
%     end
%     if isfield(m.pca,'A_Y'), 
%         Y = Y - repmat(m.pca.my,1,size(Y,2));
%         Y = Y * m.pca.A_Y; 
%     end
% end

s = struct();

if m.pcaX > 0 && m.pcaX < 1
    s.mx = mean(X);
    X = X - repmat(s.mx,size(X,1),1);
    [s.A_X,X,r_X] = pca(X);
    r_X = (cumsum(r_X)/sum(r_X));
    ncomp = find(r_X>=m.pcaX,1);
    X = X(:,1:ncomp);
    s.A_X = s.A_X(:,1:ncomp);
    if size(X,2) <= m.k
        error('K is > than the no. of principal components - increase pcaX or decrease K') 
    end
end

if m.pcaY > 0 && m.pcaY < 1
    s.my = mean(Y);
    Y = Y - repmat(s.my,size(Y,1),1);
    [s.A_Y,Y,r_Y] = pca(Y);
    r_Y = (cumsum(r_Y)/sum(r_Y));
    ncomp = find(r_Y>=m.pcaY,1);
    Y = Y(:,ncomp);
    s.A_Y = s.A_Y(:,ncomp);
    if size(Y,2) <= m.k
        error('K is > than the number of principal components - increase pcaY or decrease K')
    end
end

end