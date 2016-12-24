clc 
clear all
close all
%%%%% Number of hidden layer %%%%
format short
h_l = 1;
%%%%% Number of hidden layer neurons %%%
n_h_l = [4];
%%%%% Number of inputs %%%%
in =  2 ;
%%%%% Number of outputs %%%%
out = 2;
nodes_sizes = [in n_h_l out];
%%%% Number of Processed outputs
%%%% Suppose in , h1 , h2, out then Processed outputs(1,2,3,4) = in out_h1,out_h2, out
x= [1 0; 0 1; -1 0; 0 -1 ;0.5 0.5; -0.5 0.5 ; 0.5 -0.5; -0.5 -0.5];%; 0.5 0.25;-0.5 -0.25;-0.25 0.125;0.25 -0.125 ];
d = [1;1;1;1;0;0;0;0;];
figure
plot ( x(1:4,1),x(1:4,2),'+');
hold on
plot ( x(5:8,1),x(5:8,2),'*');
xlabel('x') % x-axis label
ylabel('y') % y-axis label
legend('Class1', 'Class2');
title('Input Data plot')
grid on;
hold off;

d = [d (1-d)];
% x = [ [0.05 0.1];x];
% d = [ [0.01 0.99];d];
% x = [0 0; 0 1 ; 1 0; 1 1];
% d = [ 0; 1; 1;0];
%  x = [1 0; 0 0 ; 0 1; 1 1];
%  d = [ 0; 1; 0;1];
for i= 1 : h_l+2
    p_out{i,1} = zeros(1,nodes_sizes(i));
end
c = 1;
% for i= 1 : h_l+1
%     for j= 1: nodes_sizes(i+1)
%         w(:,j) = c.*[ones(nodes_sizes(i),1);-1]; %%% +1 is to store bias weights
%         c= -1*c;
%     end    
%     weights{i,1} = w;
%     w=[];
% end
% w = [
%      2   -2 0  2   -2 0;
%      2   -2 0  2   -2 0;
%      1    1 1 -1   -1 -1;
%     ];
%     weights{1,1} = c.*w;
%     w=[1 1;
%        1 1;
%        1 1;
%        1 1;
%        1 1;
%        1 1;
%        1 1 ];
%     weights{2,1} = w;
w = [
     0   0    1   1;
     1   1    0   0;
    0.51 -0.51 0.51 -0.51;
    ];
    weights{1,1} = c.*w;
    w=[1 1;
       1 1;
       1 1;
       1 1;
       1 1 ];
    weights{2,1} = w;

    
%%%% Net starts %%%%
mse = Inf;
epochs = 0;
[samples,~] = size(x);
e = zeros(samples,out);
smse = 0.001;
eta = 0.5;
loss = 0;
batch = 1;

while mse > smse && epochs < 30000
%%while epochs < 30000
    for k = 1 : batch:samples
        batch = max(min(samples-k,batch),1);
        [e(k:k+batch-1,:),weights, y(k:k+batch-1,:)] = neural_net(x(k:k+batch-1,:),d(k:k+batch-1,:),weights,p_out,h_l,nodes_sizes,eta,loss);
    end
    batch = 1;
    epochs =epochs +1;

	if (loss)
    	mse = (sum(sum(e)))/(samples*out);
	else
    	mse = sum(sum(e.^2))/(samples*out);
	end
    if mod(epochs,1000) == 0
        disp(mse);
    end
end
mse


%%%%Testing
test_d = d;
test_x = 0.75.*x; %%5- 0.25.*ones(size(x));
[test_samples,~] = size(test_x);
   %% if shuffle == 1
        [sz,in]= size(test_x);
        [~,o]= size(test_d);
        s = [test_x test_d];
        s = s(randperm(sz),:);
        test_x = s(:,1:in);
        test_d = s(:,in+1:in+o);
    %%end
test_y = zeros(size(d));
for k = 1 : test_samples
        [p_out,y] = forward_path(test_x(k,:),weights,p_out,nodes_sizes,h_l,loss);
        [~,sz] = size(y);
        test_y(k,:) = y(1,1:sz-1);
end
test_y
[C_Mat]=ConfusionMatrix(test_d,test_y)
%columns train data desired output
test_x= [0.75 0.75; 0.25 0.25; 0.25 -0.25; -0.25 -0.25 ;-0.25 0.25; 0.75 0.25 ; 0.6 -0.6; 0.25 0.75];
test_d = [1;0;0;0;0;1;1;1];
test_d = [test_d (1-test_d)];
[test_samples,~] = size(test_x);
test_y = zeros(size(test_d));
for k = 1 : test_samples
        [p_out,y] = forward_path(test_x(k,:),weights,p_out,nodes_sizes,h_l,loss);
        [~,sz] = size(y);
        test_y(k,:) = y(1,1:sz-1);
end
test_y
[C_Mat]=ConfusionMatrix(test_d,test_y)
