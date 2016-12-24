function [Con_Mat] = ConfusionMatrix(d,y)
[samples,sz]= size(y);
[V, P]= max(y,[],2); %%% Index to get highest probability
[W, D]= max(d,[],2); %%% Index to get highest probability
Con_Mat= zeros(sz,sz);
for i = 1: samples
    Con_Mat(P(i),D(i))= Con_Mat(P(i),D(i))+1;
end
end