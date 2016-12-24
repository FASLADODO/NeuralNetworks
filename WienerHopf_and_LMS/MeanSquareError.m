function [mse , degree , error,pred] = MeanSquareError(w, in , out)

[samples,~] = size(out);
pred = (w' * in')';
error = out-pred;

% mean squared error
mse = sum(error.^2)/samples;

% degree of whiteness
degree = fft(error' * error);  
%%The value seen to be equal to MSE
end