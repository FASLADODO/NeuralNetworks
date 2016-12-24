function [eigv,eigvec,ld,tranformdata]=lda_space(data,sum_covar,meandata,k,o_mean)
%%%Calculating between class covariance
[~,sz]= size(k);
[~,width] = size(o_mean);
y_m =  meandata- ones(sz,width)*diag((o_mean(:,:)));
x_m =  ((k(1))).*y_m; %%% Number of samples for each class are equal
sb= x_m' *y_m;
%%%% Calculating the LDA eigen space vectors
A = sum_covar \ sb;
[D,V] = eig(A);
[eigv,I] = sort(diag(V),'descend');
eigvec = D(I,:);
pereigv= eigv.*100/sum(eigv);
if (sum(pereigv(1:2)) > 90 )
    ld = 2;
else
    ld = 3;
end
tranformdata = data*(eigvec(1:ld,:))';
end
