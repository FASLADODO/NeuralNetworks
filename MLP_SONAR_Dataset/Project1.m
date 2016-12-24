clear all
close all
clc
filename = 'data.txt';
formatSpec='%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%C';
delimiterIn = ' ';
T = readtable('data.txt','Delimiter',',','ReadVariableNames',false,'Format',formatSpec);

%%%%% load the best weights %%
% load 24293dump.mat;
% l= zeros(1,24);
% figure
% plotwb(weights{1}(1:60,:)',l',weights{1}(61,:)');
% xlabel('Input to Hidden layer Final Weights and Biases');
% %plotwb(I_weights{1}(1:60,:)',I_weights{2}(1:24,:),I_weights{1}(61,:)');
% figure
% plotwb(weights{2}(1:24,:)',[0 0]',weights{2}(25,:)');
% xlabel('Hidden to Output Layer Final Weights and Biases');
% figure

% plotwb(weights{1}(1:60,:)',[0 0]',weights{1}(61,:)');
%InTrn = InTrn1;
for times= 1 :15
%%val_sz =  round(0.1 * 2/3 * 208);
[InTrn,InTst1] = arrange(T,2); %% data generation
val_en =0;
InTrn1 = InTrn;
% f_weights=weights;
% f_Trerror=Trerror;
% f_Tsterror=Tsterror;
% f_TrC_Mat = TrC_Mat;
% f_TstC_Mat = TstC_Mat;

%%% Validation data set
if (val_en == 1)
    idx = randperm(length(InTrn1),floor(0.1*length(InTrn1)));
    val_x= InTrn1(idx,:);
    prev =0;
    idx=sort(idx);
    C=[];	
    for i=1:floor(0.1*length(InTrn1))
        C= [C;InTrn1(prev+1:idx(i)-1,:)];
            prev = idx(i);
    end
    InTrn1 = C;
    val_d = val_x(:,61);
    val_x = val_x(:,1:60);
    val_d = [val_d (1-val_d)];
    [val_samples,~] = size(val_x);
end
x = InTrn1(:,1:60);
d = InTrn1(:,61);
d = [d (1-d)];


AvgError=0;
AvgTrError =0;
MaxError = 0;
for j=1:30
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Number of hidden layer %%%%
h_l = 0;
%%%%% Number of hidden layer neurons %%%
n_h_l = [0];
%%%%% Number of inputs %%%%
[samples,in] = size(x);
%%%%% Number of outputs %%%%
[~,out] = size(d);
if (h_l)
    nodes_sizes = [in n_h_l out];
else
    nodes_sizes = [in out];
end
%%%% Number of Processed outputs
%%%% Suppose in , h1 , h2, out then Processed outputs(1,2,3,4) = in out_h1,out_h2, out

for i= 1 : h_l+2
    p_out{i,1} = zeros(1,nodes_sizes(i));
end
for i= 1 : h_l+1
    w = -0.3+ 0.6*rand(nodes_sizes(i)+1,nodes_sizes(i+1)); %%% +1 is to store bias weights
    weights{i,1} = w;
end
%weights = I_weights;
%%%% Net starts %%%%
mse = Inf;
epochs = 0;
m = 0.01; %momentum
e = zeros(samples,out);
smse = 0.001;
eta0 = 0.7;
eta = eta0; %eta learning rate 
loss = 0;
max_val_e = Inf;
val_e = 0;
val_chk = 1;
stop =  mse > smse && val_chk && epochs < 800;
%%while mse > smse && epochs < 30000
batch =5;
I_weights= weights;
% for i= 1 : h_l+1
%     weights{i,1}
% end
%while mse > smse && epochs < 15000
e=[];
y=[];
test_y=[];
shuffle =0;
while stop
    for k = 1 : batch:samples
        batch = max(min(samples-k,batch),1);
        [e(k:k+batch-1,:),weights, y(k:k+batch-1,:)] = neural_net(x(k:k+batch-1,:),d(k:k+batch-1,:),weights,p_out,h_l,nodes_sizes,eta,loss,m);
    end
    batch = 5;
    epochs =epochs +1;
    eta = eta0/(1+epochs/400);
    if (loss)
    	 mse = (sum(sum(e)))/samples;
	else
    	mse = sum(sum(e.^2))/(samples*out);
	end
	%%% Validation inputs
	if(val_en ==1 && (mod(epochs,10) == 0 ))
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
            f_weights = weights;
            val_mse =mse; 
        else
			%val_chk = 0;
		end
	end
	
    if mod(epochs,1000) == 0
        %display(mse);
    end
	stop = (mse > smse) && val_chk && epochs < 800;
    %%% To Shuffle data
    if((shuffle ==1 ) && (stop ==1))
        t =  rshuffle([x d]);
        x = t(:,1:60);
        d = t(:,61:62);
    end
end
%display ('Results\n');
% mse
% epochs

[tr_samples,~] = size(x);
tr_y = zeros(size(d));
for k = 1 : tr_samples
        [p_out,y] = forward_path(x(k,:),weights,p_out,nodes_sizes,h_l,loss);
        [~,sz] = size(y);
        tr_y(k,:) = y(1,1:sz-1);
    end
% for i= 1 : h_l+1
%     weights{i,1}
% end
%weights{4,1}
%display('Training Results');
[TrC_Mat]=ConfusionMatrix(d,tr_y);
Trerror = (trace(TrC_Mat)/sum(sum(TrC_Mat)))*100;

% columns train data input
test_x = InTst1(:,1:60);
test_d = InTst1(:,61);
test_d = [test_d (1-test_d)];
[test_samples,~] = size(test_x);
test_y = zeros(size(test_d));
for k = 1 : test_samples
        [p_out,y] = forward_path(test_x(k,:),weights,p_out,nodes_sizes,h_l,loss);
        [~,sz] = size(y);
        test_y(k,:) = y(1,1:sz-1);
end
%%% If validation enabled
if (val_en)
test_y = zeros(size(test_d));
for k = 1 : test_samples
        [p_out,y] = forward_path(test_x(k,:),weights,p_out,nodes_sizes,h_l,loss);
        [~,sz] = size(y);
        test_val(k,:) = y(1,1:sz-1);
end
[ValC_Mat]=ConfusionMatrix(test_d,test_val);
Valerror = (trace(ValC_Mat)/sum(sum(ValC_Mat)))*100;
end
% for i= 1 : h_l+1
%     weights{i,1}
% end
%weights{4,1}
%display('Testing Results');
[TstC_Mat]=ConfusionMatrix(test_d,test_y);
Tsterror = (trace(TstC_Mat)/sum(sum(TstC_Mat)))*100;
AvgError= AvgError + Tsterror;
if ( MaxError < Tsterror)
     MaxError = Tsterror;
 end
%%% If validation enabled
if((val_en) && (Valerror > Tsterror))
    if (Valerror > 80)
    s = strcat(int2str(times),strcat(int2str(round(Valerror)),'file.mat'));
    valdata = [val_x val_d(:,1)];
    %display(s);
    %%if exist(Name, s) ~= 2
    save(s,'Valerror','InTrn1','valdata','I_weights','weights','InTst1','val_mse','Trerror','TrC_Mat','ValC_Mat');
    %%end
    end
    AvgError= AvgError + (Valerror-Tsterror);
    if ( MaxError < Valerror)
     MaxError = Valerror;
    end
end
 

AvgTrError= AvgTrError + Trerror;
if (Tsterror > 85 )
   s = strcat(int2str(n_h_l),strcat(int2str(times),strcat(int2str(round(Tsterror)),'dump.mat')));
   save(s,'Tsterror','InTrn1','I_weights','weights','InTst1','mse','Trerror','TrC_Mat','TstC_Mat'); 
end
end
display(times);
 fid = fopen('run.txt','a+');
 fprintf(fid,'%d %f %f %f\n',n_h_l,MaxError,AvgError/30,AvgTrError/30);
 fclose(fid);
MaxError 
MaxError =0;
AvgError = AvgError /30
AvgError =0;
AvgTrError = AvgTrError /30
AvgTrError =0;
end


