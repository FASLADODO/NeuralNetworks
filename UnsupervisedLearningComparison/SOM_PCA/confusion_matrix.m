function confusion_matrix = confusion_matrix(output_test,output)

[samples,sz]= size(output);

[~, P]= max(output,[],2); 
[~, D]= max(output_test,[],2); 
confusion_matrix= zeros(sz,sz);

for i = 1: samples
    confusion_matrix(P(i),D(i))= confusion_matrix(P(i),D(i))+1;
end

end