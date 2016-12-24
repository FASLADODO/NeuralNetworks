function [ f , d_f ] = sigmoid(x,option)
    d_f = 0;
    if ( option == 0)
        f= 1./(1 + exp(-x));
    elseif option == 1
        f= exp(x);
    end
end
