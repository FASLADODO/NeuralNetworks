clc 
clear all
delimiterIn = '\t';
headerlinesIn = 1;
A = importdata('Sleepdata1 Desired.asc',delimiterIn,headerlinesIn);
% columns train data desired output
d = A.data;
A = importdata('Sleepdata1 Input.asc',delimiterIn,headerlinesIn);
% columns train data input
x = A.data;
val_x = x(290:315,:);
val_d = d(290:315,:);
[val_samples,~] = size(val_x);
val_en= 0;
if (val_en == 1)
	[samples,in] = size(x);
	x = [x(1:289,:);x(315:samples-1,:)];
	d = [d(1:289,:);d(315:samples-1,:)];
end
%%%%% Number of hidden layer %%%%
h_l = 3;
%%%%% Number of hidden layer neurons %%%
n_h_l = [10 8 6]
%%%%% Number of inputs %%%%
[samples,in] = size(x);
%%%%% Number of outputs %%%%
[~,out] = size(d);
nodes_sizes = [in n_h_l out];
%%%% Number of Processed outputs
%%%% Suppose in , h1 , h2, out then Processed outputs(1,2,3,4) = in out_h1,out_h2, out

for i= 1 : h_l+2
    p_out{i,1} = zeros(1,nodes_sizes(i));
end
for i= 1 : h_l+1
    w = -1+ 2*rand(nodes_sizes(i)+1,nodes_sizes(i+1)); %%% +1 is to store bias weights
    weights{i,1} = w;
end
% w = [ 1 1; -1 -1; -1 -1];
% weights{1,1} = w;
% w = [ 2; -2; -1];
% weights{2,1} = w;
%%%% Net starts %%%%
mse = Inf;
epochs = 0;

e = zeros(samples,out);
smse = 0.01;
eta0 = 0.8
eta = eta0;
loss = 0;
max_val_e = Inf;
val_e = 0;
val_chk = 1;
stop =  mse > smse && val_chk && epochs < 30000;
%%while mse > smse && epochs < 30000
batch =10;
for i= 1 : h_l+1
    weights{i,1}
end
%while mse > smse && epochs < 15000
while stop
    for k = 1 : batch:samples
        batch = max(min(samples-k,batch),1);
        [e(k:k+batch-1,:),weights, y(k:k+batch-1,:)] = neural_net(x(k:k+batch-1,:),d(k:k+batch-1,:),weights,p_out,h_l,nodes_sizes,eta,loss);
    end
    batch = 10;
    epochs =epochs +1;
    eta = eta0/(1+epochs/5000);
    if (loss)
    	mse = (sum(sum(e)))/samples;
	else
    	mse = sum(sum(e.^2))/(samples*out);
	end
	%%% Validation inputs
	if(val_en ==1)
		for k = 1 : val_samples
			[~,val_y] = forward_path(val_x(k,:),weights,p_out,nodes_sizes,h_l,loss);
			[~,sz] = size(val_y);
			test_y(k,:) = val_y(1,1:sz-1);
		end
		if (loss)
			err = (-1).*log(test_y(:,1:sz-1));
			val_e = (sum(sum(err)))/samples;
		else
			err = val_d - test_y(:,1:sz-1);
			val_e = sum(sum(err.^2))/(samples*out);
		end
		if (( max_val_e -val_e)  >= 0.000)
			max_val_e = val_e;
		else
			val_chk = 0;
		end
	end
	
    if mod(epochs,1000) == 0
        %display(mse);
    end
	stop = (mse > smse) && val_chk && epochs < 20000;
end
display ('Results\n');
mse



%%%% Testing Phase %%%%
% test_x = x;
% [test_samples,~] = size(x);
% test_y = zeros(size(d));
% for k = 1 : test_samples
%         [p_out,y] = forward_path(test_x(k,:),weights,p_out,nodes_sizes,h_l,loss);
%         [~,sz] = size(y);
%         test_y(k,:) = y(1,1:sz-1);
% end
% test_y
% [C_Mat]=ConfusionMatrix(d,test_y)
% error = trace(C_Mat)/sum(sum(C_Mat))



delimiterIn = '\t';
headerlinesIn = 1;
A = importdata('Sleepdata2 Desired.asc',delimiterIn,headerlinesIn);
% columns train data desired output
test_d = A.data;
A = importdata('Sleepdata2 Input.asc',delimiterIn,headerlinesIn);
% columns train data input
test_x = A.data;
[test_samples,~] = size(test_x);
test_y = zeros(size(test_d));
for k = 1 : test_samples
        [p_out,y] = forward_path(test_x(k,:),weights,p_out,nodes_sizes,h_l,loss);
        [~,sz] = size(y);
        test_y(k,:) = y(1,1:sz-1);
end
for i= 1 : h_l+1
    weights{i,1}
end
%weights{4,1}
[C_Mat]=ConfusionMatrix(test_d,test_y)
error = (trace(C_Mat)/sum(sum(C_Mat)))*100