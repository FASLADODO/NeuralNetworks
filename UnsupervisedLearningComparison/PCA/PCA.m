%% PCA code for custering

function result = PCA(output,data,output1_test,input_test, odim)

%% for train data

[N, n] = size(data);
% calculating autocorrelation matrix
A = zeros(n);
me = zeros(1,n);

for i=1:n
    me(i) = mean(data(isfinite(data(:,i)),i));
    data(:,i) = data(:,i) - me(i); 
end

for i=1:n 
    for j=i:n
        c = data(:,i).*data(:,j);
        c = c(isfinite(c));
        A(i,j) = sum(c)/length(c);
        A(j,i) = A(i,j);
    end
end

% eigenvectors, sort them according to eigenvalues, and normalize
[V,S]   = eig(A);
eigval  = diag(S);
[y,ind] = sort(abs(eigval)); 
eigval  = eigval(flipud(ind));
V       = V(:,flipud(ind)); 

for i=1:odim
    V(:,i) = (V(:,i) / norm(V(:,i)));
end

% take only odim first eigenvectors
%%% 84.5 Accuracy for subspace 3 and 4 a2 == 1
%%% 85.5 Accuracy for subspace 3 and 5 (1 -14.5) a2 == 2
V = [V(:,3)  V(:,5)];
D = abs(eigval)/sum(abs(eigval));
D = [D(2) ; D(3)];


% project the data using odim first eigenvectors
X = data*V;

% cluster using k-means
[idx,C,sumd,D] = kmeans(X,2);

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3)
%%legend('Cluster 1','Cluster 2','Centroids','Location','NW')
title 'Clusters for Species Class and Centroids'
%%hold off

Aoutput = [];
doutput = [];
for k = 1:N
    a1 = output(k,1);
    a2 = idx(k,1);
    if(a1 == 1)
        b1 = 0;
        c1 = 1;
    else
        b1 = 1;
        c1 = 0;
    end
    Aoutput = [Aoutput; [c1 b1]];
    if(a2 == 1)
        b1 = 0;
        c1 = 1;
    else
        b1 = 1;
        c1 = 0;
    end
    doutput = [doutput; [c1 b1]];
end
%for train data
con = confusion_matrix(doutput,Aoutput)

result1 = trace(con)*100/sum(sum(con))

%% for test data

[N, n] = size(input_test);
% calculating autocorrelation matrix
A = zeros(n);
me = zeros(1,n);

for i=1:n
    me(i) = mean(input_test(isfinite(input_test(:,i)),i));
    input_test(:,i) = input_test(:,i) - me(i); 
end

for i=1:n 
    for j=i:n
        c = input_test(:,i).*input_test(:,j);
        c = c(isfinite(c));
        A(i,j) = sum(c)/length(c);
        A(j,i) = A(i,j);
    end
end

% eigenvectors, sort them according to eigenvalues, and normalize
[V,S]   = eig(A);
eigval  = diag(S);
[y,ind] = sort(abs(eigval)); 
eigval  = eigval(flipud(ind));
V       = V(:,flipud(ind)); 

for i=1:odim
    V(:,i) = (V(:,i) / norm(V(:,i)));
end

% take only odim first eigenvectors
%%% 84.5 Accuracy for subspace 3 and 4 a2 == 1
%%% 85.5 Accuracy for subspace 3 and 5 (1 -14.5) a2 == 2
V = [V(:,3)  V(:,5)];
D = abs(eigval)/sum(abs(eigval));
D = [D(2) ; D(3)];


% project the data using odim first eigenvectors
testX = input_test*V;

Edist = pdist2(testX,C,'euclidean');
[s, ~]= size(Edist);
for i=1 : s
    if ( Edist(i,1) > Edist(i,2))
        Edist(i,2) = 1;
        Edist(i,1) = 0;
    else
        Edist(i,2) = 0;
        Edist(i,1) = 1;
    end
end
hold on 
plot(testX(Edist(:,1)==0,1),testX(Edist(:,1)==0,2),'go','MarkerSize',2)
hold on
plot(testX(Edist(:,1)==1,1),testX(Edist(:,1)==1,2),'co','MarkerSize',2)
hold off
legend('Cluster 1','Cluster 2','Centroids','Test Cluster 2',...
'Test Cluster 1','Location','NW')
desired_test = [];
for k = 1:s
    a1 = output1_test(k,1);
    a2 = idx(k,1);
    if(a1 == 1)
        b1 = 0;
        c1 = 1;
    else
        b1 = 1;
        c1 = 0;
    end
    desired_test = [desired_test; [c1 b1]];
end
con = confusion_matrix(desired_test,Edist)

result = trace(con)*100/sum(sum(con))
end
