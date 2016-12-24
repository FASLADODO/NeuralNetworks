%% Wiener filter to find weights directly

function [ R , weights,pred ] = WienerFilter(y, x, L, lambda)
[Lx,~] = size(x);
L= L+1;
[Lx,~] = size(x);

rxx = xcorr(x,L-1,'biased')/Lx;      % autocorrelation sequence over (N-1) lags
R   = toeplitz(rxx(L:-1:1),rxx(L:2*L-1));
%                            covariance matrix
   % cross-correlation vector
[p,~] = size(R);
% If Autocorrelation is not full rank so inverse won't exist
% So Adding a Identity with lambda multiplier to make it full rank 
if rank(R) < p
	R = R + lambda * eye(p);
end
rdx = xcorr(y,x,L-1,'biased')/Lx;    % cross-correlation sequence over (N-1) lags
P  = (rdx(L:2*L-1));
weights = inv(R)*P;           % Wiener-Hopf solution 
pred = filter(weights,1,x);
end