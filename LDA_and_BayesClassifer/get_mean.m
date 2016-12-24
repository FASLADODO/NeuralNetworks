function [mean_matrix] = get_mean(classes,data,k)
[~,sz]= size(k);
[~,width] = size(data);
mean_matrix = zeros(sz,width);
offsets= zeros(1,2);
offsets(1) = 1;
for i =1 : sz 
    offsets(2) = sum(k(1:i));
    for j =1: width
       mean_matrix(i,j)= mean(data(offsets(1):offsets(2),j));
    end
    offsets(1) = offsets(2)+1;
end
end