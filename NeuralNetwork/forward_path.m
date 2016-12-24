function [p_out,y] = forward_path(x,weights,p_out,nodes_sizes,h_l,loss)
[sz,~] = size(x);
p_out{1,1} = [x ones(sz,1)];

%%% Forward Path %%%%%
for i = 2:h_l+2
    w = weights{i-1,1};
    in = p_out{i-1,1};
    out= zeros(sz,nodes_sizes(i));
    for j = 1: nodes_sizes(i)
        if ( i < h_l+2)
            [out(:,j),~] = sigmoid(in*w(:,j),0);
        else
            if (loss)
				out(:,j) = in*w(:,j);
			else
				[out(:,j),~] = sigmoid(in*w(:,j),0);
			end	
        end
    end
	if (loss)
	    if(i == h_l+2) %% SoftMax Layer 
	        out = out - max(out).*ones(sz,nodes_sizes(i));
	        out = exp(out);
	        out = (1./(sum(out))).*out;
	    end
	end
    p_out{i,1} = [out ones(sz,1)];
    in =[];
    w =[];
end
y = p_out{h_l+2,1};
end