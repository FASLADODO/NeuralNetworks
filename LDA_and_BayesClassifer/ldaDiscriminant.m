function [ G1, G2, G3, o,tmcovar,c] = ldaDiscriminant(classes, trans_data, k, X)
[ transformMean ] = get_mean(classes,trans_data,k);
tranformcovar = cov(trans_data);
[~,sz]= size(k);
for i = 1 : sz
    tcovar(:,:,i) = cov(trans_data((1):k(1),:));
    tmcovar(:,:,i) = tcovar(:,:,i)\(transformMean(i,:))';
    c(:,:,i)= log(1/sz)-1/2.*(transformMean(i,:)* tmcovar(:,:,i));
    o(i) = X * tmcovar(:,:,i) + c(:,:,i);
end
eq1 = simplify ( o(1));
eq2 = simplify ( o(2));
eq3 = simplify ( o(3));
G1 = simplify(eq1 -eq2);
G2 = simplify(eq2 -eq3);
G3 = simplify(eq3 -eq1);
end