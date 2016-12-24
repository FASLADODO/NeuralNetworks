%% clean up
close all
clear all
clc


%% Predict a curve to fit the following relation of the desired output
% y(n) = y(n-1)/(0.1 + (y(n-4)^2) + sin ( x(i-3)))^2
% this is our time series data
samples = 2000;
% generate white gaussian noise with unit variance response
g = randn(samples,1);

x = zeros(samples +4,1);
y = zeros(samples +4,1);
x = cat ( 1 , zeros(4,1),g(1:end));
t = zeros (samples +4,1);
for i = 5: 1: samples +4
    t(i) = (sin ( x(i-3)))^2;
    y(i) = (y(i-1)/(0.1 + y(i-4)^2))+ t(i) ;
end

x = x(5:samples +4);
y = y(5:samples +4);

test_samples = 0.1 *samples;
train_samples = 0.9 *samples;
in_train = x(1:train_samples);
in_test = x(train_samples+1:samples);
[sz,~] = size(in_train);
bias_train = ones(sz,1);
out_train = y(1:train_samples);
out_test = y(train_samples+1:samples);
[tsz,~] = size(in_test);
bias_test = ones(tsz,1);
figure
plot(y);
title('Desired Input');
figure
plot(x);
title('Desired Output');
%%%%% Arrange filter order %%%%%%%%
[in_train5] = rearrange(in_train,5,sz);
[in_train15]= rearrange(in_train,15,sz);
[in_test5] = rearrange(in_test,5,tsz);
[in_test15]= rearrange(in_test,15,tsz);
%% Wiener filter implementation to find the w_tracks

lambda = 0.001;
%% Filter order 5
[R5 weights_wiener5 ,pred5] = WienerFilter(out_train, in_train,5,lambda);
[R15 weights_wiener15 ,pred15] = WienerFilter(out_train, in_train,15,lambda);
figure,
hold on 
plot(out_train,'+');
plot(pred5,'o');
plot(pred15,'*');
xlabel('Input') % x-axis label
ylabel('Desired and Predicted Responses)') % y-axis label
legend('Desired', 'Wiener FIR Order=5', 'Wiener FIR Order=15' );
title('Wiener Filter Variation')
grid on;
hold off;

figure 
subplot(2,1,1);
plot(weights_wiener5);
title('Wiener Filter Impulse Response Order 5');
grid on
subplot(2,1,2);
plot(weights_wiener15);
title('Wiener Filter Impulse Response Order 15');
grid on
%fitcurve(weights_1, in_test , out_test);
W_error5 = out_train-pred5;
W_error15 = out_train-pred15;
% mean squared error
W_mse5 = sum(W_error5.^2)/samples
W_mse15 = sum(W_error15.^2)/samples
H=spectrum.periodogram;
figure
subplot(2,1,1);
plot(psd(H,W_error5));
subplot(2,1,2);
plot(psd(H,W_error15));
%mse_1
%%% eta related terms initialized
R = in_train'*in_train;
[s,~] = size(in_train); 
eta0 = 0.05;%eta0 be 0.05 to increase the convergence rate
e = eig(R); %% eigenvalues of input autocorrelation
trace = trace(R); %% trace of input autocorrelation
% eigen value 'e' of input auto correlation matrix to find range and speed
% of convergence of eta
eta_max = 2/(max(e));
eta = eta0/ trace;

%% On-line iterative LSM for FIR filter order 5
 iterations = 100;  
order = 5;
weights_online5 = zeros(20,order+2,1);
w_tracks = zeros(20,iterations*sz,order+2);
Jmin = zeros(20,iterations,1);
eta_samples = 20;
figure,
for i = 1:eta_samples
    k = mod(i-1,20)+1;
    eta_val_online5(k) = eta*(10+i);
    % inverse of time of adaptation
    timeconstantofadaptation_5(k) = (1/(eta_val_online5(k)*min(e)));
    misadjustment_5(k) = eta_val_online5(k)*trace;
    [weights_online5,  count , w_tracks(k,:,:) , Jmin(k,:,:) , ~] =...
        onlineLMS(order,sz, in_train5 , out_train , bias_train , iterations , eta_val_online5(k));
    [mse_train_5(k,:,:), degreeOfWhiteness_2, error_5(k,:,:),~] = MeanSquareError(weights_online5, [in_train5 bias_train] , out_train);
    [mse_test_5(k,:,:), degreeOfWhiteness_2, terror_5(k,:,:),~] = MeanSquareError(weights_online5, [in_test5 bias_test] , out_test);
%     if ( mse_train_online(k) < 0.01)
%         break;
%     end
end
for i=1:20
    plot(1:iterations,Jmin(i,:,:),'-')
    hold on
end
title('Online Learning Curve (FIR order = 5)')
xlabel('Iterations') % x-axis label
ylabel('Jmin') % y-axis label
legend(num2str(eta_val_online5(1)) , num2str(eta_val_online5(2)) , num2str(eta_val_online5(3))...
    , num2str(eta_val_online5(4)) , num2str(eta_val_online5(5)) , ...
    num2str(eta_val_online5(6)) , num2str(eta_val_online5(7)) , num2str(eta_val_online5(8))...
    , num2str(eta_val_online5(9)) , num2str(eta_val_online5(10)) , ...
    num2str(eta_val_online5(11)) , num2str(eta_val_online5(12)) , num2str(eta_val_online5(13))...
    , num2str(eta_val_online5(14)) , num2str(eta_val_online5(15)) , ...
    num2str(eta_val_online5(16)) , num2str(eta_val_online5(17)) , num2str(eta_val_online5(18))...
    , num2str(eta_val_online5(19)) , num2str(eta_val_online5(20)) )
grid on;
hold off;

% eta value for minimum Jmin
[Jmin_5,I5] = min(mse_train_5);
display('Online FIR =5 details start');
Jmin_5
eta_val_online5(I5)
timeconstantofadaptation_5(I5) 
speedofadaptation_5=iterations/timeconstantofadaptation_5(I5)
misadjustment_5(I5)
mse_test_5
display('Online FIR =5 details over')
figure,
for i=1:order+2
plot(1:count-1,w_tracks(I5,:,i),'--');
hold on;
end
title(strcat('Plot of  Weight tracks eta=',num2str(eta_val_online5(I5))))
xlabel('Iterations') % x-axis label
ylabel('weights') % y-axis labelg
legend('W0','W1','W2','W3','W4','W5','Bias')
grid on;
hold off;

%% On-line iterative LSM for FIR filter order 15

order = 15;

% eigen value 'e' of input auto correlation matrix to find range and speed
% of convergence of eta
eta_max = 2/(max(e));
eta = eta0/ trace;
weights_online15 = zeros(20,order+2,1);
w_tracks = zeros(20,iterations*sz,order+2);
Jmin = zeros(20,iterations,1);
    
figure,
for i = 1:eta_samples
    k = mod(i-1,20)+1;
    eta_val_online15(k) = eta*(10+i);
    % inverse of time of adaptation
    timeconstantofadaptation_15(k) = (1/(eta_val_online15(k)*min(e)));
    misadjustment_15(k) = eta_val_online15(k)*trace;
    [weights_online15,  count , w_tracks(k,:,:) , Jmin(k,:,:) , ~] =...
        onlineLMS(order,sz, in_train15 , out_train , bias_train , iterations , eta_val_online15(k));
    [mse_train_15(k,:,:), degreeOfWhiteness_2, error_15(k,:,:),~] = MeanSquareError(weights_online15, [in_train15 bias_train] , out_train);
    [mse_test_15(k,:,:), degreeOfWhiteness_2, terror_15(k,:,:),~] = MeanSquareError(weights_online15, [in_test15 bias_test], out_test);
    if ( mse_train_15(k) < 0.001)
        break;
    end
end
for i=1:20
    plot(1:iterations,Jmin(i,:,:),'-')
    hold on
end
title('Online Learning Curve (FIR order = 15)')
xlabel('Iterations') % x-axis label
ylabel('Jmin') % y-axis label
legend(num2str(eta_val_online15(1)) , num2str(eta_val_online15(2)) , num2str(eta_val_online15(3))...
    , num2str(eta_val_online15(4)) , num2str(eta_val_online15(5)) , ...
    num2str(eta_val_online15(6)) , num2str(eta_val_online15(7)) , num2str(eta_val_online15(8))...
    , num2str(eta_val_online15(9)) , num2str(eta_val_online15(10)) , ...
    num2str(eta_val_online15(11)) , num2str(eta_val_online15(12)) , num2str(eta_val_online15(13))...
    , num2str(eta_val_online15(14)) , num2str(eta_val_online15(15)) , ...
    num2str(eta_val_online15(16)) , num2str(eta_val_online15(17)) , num2str(eta_val_online15(18))...
    , num2str(eta_val_online15(19)) , num2str(eta_val_online15(20)) )
grid on;
hold off;

% eta value for minimum Jmin
display('Online FIR =15 details over')
[Jmin_15,I15] = min(mse_train_15);
Jmin_15
eta_val_online15(I15)
timeconstantofadaptation_15(I15) 
speedofadaptation_15=iterations/timeconstantofadaptation_15(I15)
misadjustment_15(I15)
mse_test_15
display('Online FIR =15 details over')
figure
for i=1:order+2
plot(1:count-1,w_tracks(I15,:,i),'--');
hold on;
end
title(strcat('Plot of  Weight tracks eta=',num2str(eta_val_online15(I15))))
xlabel('Iterations') % x-axis label
ylabel('weights') % y-axis labelg
legend('W0','W1','W2','W3','W4','W5','W6','W7',...
'W8','W9','W10','W11','W12','W13','W14','W15','Bias');
grid on;
hold off;

%% Batch-line iterative LSM
batch = 5;
order = 5;
eta0 = 0.05;
% eigen value 'e' of input auto correlation matrix to find range and speed
% of convergence of eta
iterations = 250;
eta_max = 2/(max(e));
eta = eta0/ trace;
figure,
weights_batch = zeros(20,order+2,1);
w_tracks_batch = zeros(20,iterations*sz/batch,order+2);
Jmin_batch = zeros(20,iterations,1);
    
figure,
for i = 1:eta_samples
    k = mod(i-1,20)+1;
    eta_val_batch(k) = eta*(i);
    % inverse of time of adaptation
    timeconstantofadaptation_batch(k) = (1/(eta_val_batch(k)*min(e)));
    misadjustment_batch(k) = eta_val_batch(k)*trace;
    [weights_batch,  count , w_tracks_batch(k,:,:) , Jmin_batch(k,:,:) , ~] =...
        batchLMS(order,sz, in_train5 , out_train , bias_train , iterations , eta_val_batch(k),batch);
    [mse_train_batch(k,:,:), ~, error_batch(k,:,:),~] = MeanSquareError(weights_batch, [in_train5 bias_train] , out_train);
    [mse_test_batch(k,:,:), ~, terror_batch(k,:,:),~] = MeanSquareError(weights_batch, [in_test5 bias_test] , out_test);
    if ( mse_train_batch(k) < 0.001)
        break;
    end
end
for i=1:20
    plot(1:iterations,Jmin_batch(i,:,:),'-')
    hold on
end

title('Batch Learning Curve FIR =5')
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

figure
[Jmin_B,IB] = min(mse_train_batch);
Jmin_B
eta_val_batch(IB)
timeconstantofadaptation_batch(IB) 
speedofadaptation_batch=iterations/timeconstantofadaptation_batch(IB)
misadjustment_batch(IB)
mse_test_batch
for i=1:order+2
plot(1:count,w_tracks_batch(IB,:,i),'--');
hold on;
end
title(strcat('Plot of Batch Weight tracks eta=',num2str(eta_val_batch(IB))))
xlabel('Iterations') % x-axis label
ylabel('weights') % y-axis labelg
legend('W0','W1','W2','W3','W4','W5','Bias');
grid on;
hold off;