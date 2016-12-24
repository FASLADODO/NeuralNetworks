function [w,k, w_tracks , J , corr_coeff] = batchLMS(order , N , x , y , bias , max_iter , lambda , batch)        
order = order +1;
norm =1; 
if ( order > 1 )
    norm = order;
end
order =1;
% w_tracks are initialized to 1 to not ignore first sample
[~,samples] = size(x);
w = zeros(1, samples+1);
%%% input with bias and output matrix are initialized
[N,~] = size(y);
x = [x bias];
%% J (MSE) ,w_tracks ( Weight tracks) and gradJ initialized
J = zeros(max_iter, 1);
w_tracks = zeros(floor(N*max_iter/batch), (samples+1));
gradJ = zeros(samples+1,batch);
k = 0;

for iter = 1 : max_iter
    for i = order :batch: N+order-1
        for j = i:batch+i-1
        u = x(i:-1:i-order+1 , :);
        e = y(i:-1:i-order+1 , :) - (1/order).*(u * w');
        gradJ(:,j) = u' * e;
        end
        Javg = sum(gradJ')/length(gradJ);
        w = w  + (lambda* Javg);
        
        k = k + 1;
        w_tracks(k,:) = w;
    end
    [J(iter),~,~,~] = MeanSquareError(w', x, y);
end

ymean=(N)*(mean(y)^2);
corr_coeff = sqrt((w*x'*y - ymean)/(y'*y - ymean));

w = w';
end