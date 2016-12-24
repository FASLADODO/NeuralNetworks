function [Con_Mat] = ConfusionMatrix( tranformdata,w,c,classes,k)
[~,sz]= size(k);
[samples, ~] = size(tranformdata);

for i = 1 : sz
    error(:,i) = tranformdata * w(:,:,i) + c(:,:,i).*ones(samples,1);
end
[V, P]= max(error,[],2);
Con_Mat= zeros(sz,sz);
for i = 1: samples
    Con_Mat(P(i),classes(i))= Con_Mat(P(i),classes(i))+1;
end
end