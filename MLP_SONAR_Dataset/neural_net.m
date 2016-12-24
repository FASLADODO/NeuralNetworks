function [e, weights, y] = neural_net(x,d,weights,p_out,h_l,nodes_sizes,eta,loss,m)
% sse = 0
% k =1;
% eta =1;
[p_out,y] = forward_path(x,weights,p_out,nodes_sizes,h_l,loss);
%%%final layer output
%%y = p_out{h_l+2,1};
[P,sz] = size(y);
%%% Cross entropy calculation for softmax classification
if (loss)
	e =-log(y(:,1:sz-1)); 
	% %% if sigmoid/softmax out be f then the derivative is f(1-f)
	delta = (d - y(:,1:sz-1))';
else
	e = d - y(:,1:sz-1);
	%% if sigmoid out be f then the derivative is f(i-f)
	d_f = y(:,1:sz-1).*(ones(P,sz-1)-y(:,1:sz-1));
	delta = e .* d_f;
end
for i=1:h_l+1
  prev_dw{i} = zeros(size(weights{i})); % prev_dw starts at 0
end 
%%%% BackPropogation %%%%%
for i =h_l+1:-1:1
    y =p_out{i,1}; %% out j layer
    sum_dw{i}= (eta.*(delta'*y))'; %%% Oj * DELTAk
    if (i >1)
        [~,sz] = size(y);
        d_f = y(:,1:sz-1).*(ones(P,sz-1)-y(:,1:sz-1));
        delta = d_f.* (weights{i,1}(1:sz-1,:)*delta')';
    end
end 
%%% Weight updates online training
for i =1:h_l+1
    prev_dw{i} = (sum_dw{i}./P) + m.*prev_dw{i};
    %%sum_dw{i} = zeros(size(w{i}));
    weights{i,1} = weights{i,1} + prev_dw{i};
end
% weights{1,1}
% weights{2,1}
%%for i=2:
[~,sz] = size(p_out{h_l+2,1});
y = p_out{h_l+2,1}(:,1:sz-1);
end