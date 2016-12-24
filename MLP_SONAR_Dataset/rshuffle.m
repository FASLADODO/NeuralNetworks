function out = rshuffle(in)
out= in(randperm(length(in)),:);
end