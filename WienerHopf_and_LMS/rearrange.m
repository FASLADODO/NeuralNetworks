function [out] = rearrange(in,order,sz)
out= in;
for i =1: order
    t = [zeros(i,1);in(1:sz-i,1)];
    out= [t out];
end
end