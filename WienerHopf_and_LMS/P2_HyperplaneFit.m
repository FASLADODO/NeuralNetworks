%% clean up
close all
clear all
clc

% order of FIR filter
order = 0;
iterations = 100;
batch = 10;

% this is our constant muti dimensional data
% this code can be easily modified for other data

dimension = 13;

filename = 'Bodyfat1 desired.asc';
delimiterIn = ' ';
A = importdata(filename,delimiterIn,1);
out_train = A.data;

[sz , ~] = size(out_train);
bias_train = ones(sz,1);

in_train = zeros(sz,dimension);

filename = 'Bodyfat2 desired.asc';
A = importdata(filename,delimiterIn,1);
out_test = A.data;

[sz , ~] = size(out_test);
in_test = zeros(sz,dimension);

filename = 'Bodyfat1 Input.asc';
A = importdata(filename);
for i = 1 : dimension
    in_train(:,i) = A.data(:,i);
end

filename = 'Bodyfat2 Input.asc';
A = importdata(filename);
for i = 1 : dimension
    in_test(:,i) = A.data(:,i);
end

[sz , ~] = size(out_train);
[tsz,~] = size(in_test);
bias_test = ones(tsz,1);
in_test = [in_test bias_test];
%% eta related terms initialized
eta_samples = 100;
eta0 = 0.5;%eta0 be 0.5 to increase the convergence rate
R = (in_train' * in_train);%% input autocorrelation
e = eig(R); %% eigenvalues of input autocorrelation
trace = trace(R); %% trace of input autocorrelation
% eigen value 'e' of input auto correlation matrix to find range and speed
% of convergence of eta
eta_max = 2/(max(e));
eta = (eta0/ trace);

% max uterations according to eta too large so un-realistic iteration value  
%iterations = 1/(4* eta * min(e));
eta_samples = eta_samples;
%% On-line iterative LSM
    
w_tracks = zeros(20,iterations*sz,dimension+1);
Jmin = zeros(20,iterations,1);
%PredY = zeros(eta_samples,iterations,1);
figure,
for i = 1:eta_samples
    k = mod(i-1,20)+1;
    eta_val_online(k) = eta*(i);
    [weights_online, count, w_tracks(k,:,:) , Jmin(k,:,:) , corr_coeff_2(k,:)] =...
        onlineLMS(order,sz, in_train , out_train , bias_train , iterations , eta_val_online(k));
    [mse_test_online(k,:,:), degreeOfWhiteness_online, terror_2(k,:,:),OPredY(k,:,:)] = MeanSquareError(weights_online, in_test , out_test);
    [mse_train_online(k,:,:), degreeOfWhiteness_online, error_2(k,:,:),~] = MeanSquareError(weights_online, [in_train bias_train] , out_train);
    if ( mse_train_online(k) < 0.0001)
        break;
    end
end
for i=1:20
    plot(1:iterations,Jmin(i,:,:),'-')
    hold on
end
title('Online Training Learning Curve')
xlabel('Iterations') % x-axis label
ylabel('Jmin') % y-axis label
legend(num2str(eta_val_online(1)) , num2str(eta_val_online(2)) , num2str(eta_val_online(3))...
   , num2str(eta_val_online(4)) , num2str(eta_val_online(5)) , ...
   num2str(eta_val_online(6)) , num2str(eta_val_online(7)) , num2str(eta_val_online(8))...
   , num2str(eta_val_online(9)) , num2str(eta_val_online(10)) , ...
   num2str(eta_val_online(11)) , num2str(eta_val_online(12)) , num2str(eta_val_online(13))...
   , num2str(eta_val_online(14)) , num2str(eta_val_online(15)) , ...
   num2str(eta_val_online(16)) , num2str(eta_val_online(17)) , num2str(eta_val_online(18))...
   , num2str(eta_val_online(19)) , num2str(eta_val_online(20)) )

grid on;
hold off;


display('Online LMS details over')
[Jmin_O,IO] = min(mse_train_online);
Jmin_O
eta_val_online(IO)
mse_test_online(IO)
corr_coeff_2(IO,:,:)
figure 
hold on
plot(out_test,'*');
plot(OPredY(IO,:,:),'+');
title('Online Predicted Response for Test Data')
legend('Desired', 'Predicted Response' );
grid on
hold off

display('Online LMS details over')
%% Batch iterative LSM
eta_samples = 100;
eta_max = 2/(max(e));
eta = (eta0/ trace);

% max uterations according to eta too large so un-realistic iteration value  
%iterations = 1/(4* eta * min(e));
eta_samples = eta_samples;
Jmin = zeros(20,iterations,1);
figure,
for i = 1:eta_samples
    k = mod(i-1,20)+1;
    eta_val_batch(k) = eta*(300+i);
    [weights_batch,count, ~ , Jmin(k,:,:) , corr_coeff_batch(k,:)] =...
        batchLMS(order,sz, in_train , out_train , bias_train , iterations , eta_val_batch(k) , batch);
    [mse_test_batch(k,:,:), degreeOfWhiteness_batch, terror_3(k,:,:),BPredY(k,:,:)] = MeanSquareError(weights_batch, in_test , out_test);
    [mse_train_batch(k,:,:), degreeOfWhiteness_batch, error_3(k,:,:),~] = MeanSquareError(weights_batch, [in_train bias_train] , out_train);
    if ( mse_train_online(k) < 0.0001)
        break;
    end
end
for i=1:20
    plot(1:iterations,Jmin(i,:,:),'-')
    hold on
end
title('Batch Training Learning Curve')
xlabel('Iterations') % x-axis label
ylabel('Jmin') % y-axis label
legend(num2str(eta_val_batch(1)) , num2str(eta_val_batch(2)) , num2str(eta_val_batch(3))...
   , num2str(eta_val_batch(4)) , num2str(eta_val_batch(5)) , ...
   num2str(eta_val_batch(6)) , num2str(eta_val_batch(7)) , num2str(eta_val_batch(8))...
   , num2str(eta_val_batch(9)) , num2str(eta_val_batch(10)) , ...
   num2str(eta_val_batch(11)) , num2str(eta_val_batch(12)) , num2str(eta_val_batch(13))...
   , num2str(eta_val_batch(14)) , num2str(eta_val_batch(15)) , ...
   num2str(eta_val_batch(16)) , num2str(eta_val_batch(17)) , num2str(eta_val_batch(18))...
   , num2str(eta_val_batch(19)) , num2str(eta_val_batch(20)) )
grid on;
hold off;
display('Batch LMS details')
[Jmin_B,IB] = min(mse_train_batch);
Jmin_B
eta_val_batch(IB)
mse_test_batch(IB)
corr_coeff_batch(IB,:,:)
figure 
hold on
plot(out_test,'*');
plot(BPredY(IO,:,:),'+');
title('Batch Predicted Response for Test Data')
legend('Desired', 'Predicted Response' );
grid on
hold off
display('Batch LMS details over')