function [scatter_cov_matrix,covariance] = within_class_scatter(classes,data,k,mean_matrix)
[~,sz]= size(k);
[~,width] = size(data);
offsets= zeros(1,2);
offsets(1) = 1;
for i =1 : sz 
    offsets(2) = sum(k(1:i));
    for j =1: width
       x_m =  data(offsets(1):offsets(2),:)- ones(k(i),width)*diag((mean_matrix(i,:)));
       covariance(:,:,i)= x_m' *x_m;
    end
    offsets(1) = offsets(2)+1;
end
scatter_cov_matrix = (1/(k(1)-1)).*(sum(covariance,3));
end




