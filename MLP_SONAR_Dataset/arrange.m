function [InTrn,InTst] = arrange (T,idx)
In = table2array(T(:,1:60));
C = table2array(T(:,61));
[sz,~] = size(C);
for i=1:sz
class(i,1) =cast((C(i,1) == 'R'),'double');
end
In = [In class];
%We know we have 208 samples in which starting 97 samples are of Rock
%While the next 111 samples are of mines so the data arrangement ahead
% is assuming this premises.
%2/3 Train and 1/3 Test sample
T=[];
class = [];
C= [];

%% get input dimesion plot
% for j=1:12
% figure,
% set(gca,'YTick',[]);
% set(gca,'XTick',[]);
% for i=1:5
%  subplot(5,1,i);
%  f= histogram(In(:,5*(j-1)+i));
%  [p,xi] = ksdensity(f.Values);
%  subplot(5,1,i);
%  plot(xi,p);
%  legend(strcat('W',int2str(5*(j-1)+i)));
% end
% end


if ( idx == 1)
    display('First data Sample');
    InTrn = [In(1:round((2/3)*97),:);In(98:97+round((2/3)*111),:)];
    InTst = [In(round((2/3)*97)+1:97,:);In(97+round((2/3)*111)+1:208,:)];
elseif (idx ==2)
    display('Second data Sample');
    In1= In(randperm(length(In(1:97,:))),:);
    In2= In(randperm(length(In(98:208,:)))+97,:);
    InTrn = [In1(1:round((2/3)*97),:);In2(1:round((2/3)*111),:)];
    InTst = [In1(round((2/3)*97)+1:97,:);In2(round((2/3)*111)+1:111,:)];
else
    display('Third data Sample');
    tr = 1; ts = 1;
    for i=1:sz
        if ( mod(i,3) == 0)
            InTst(ts,:) = In(i,:);
            ts = ts+1;
        else
            InTrn(tr,:) = In(i,:);
            tr=tr+1;
        end
    end
end

end