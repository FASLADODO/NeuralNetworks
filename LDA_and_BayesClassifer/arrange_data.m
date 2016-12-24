% load data
function [class, data , k,t_class, t_data , t_k] = arrange_data(f_desiredout,f_input,test_percent)

delimiterIn = '\t';
headerlinesIn = 1;
A = importdata(f_desiredout,delimiterIn,headerlinesIn);
% columns represent class 2, class 3 and class 1 from left
classes = A.data;
A = importdata(f_input,delimiterIn,headerlinesIn);
% columns represent Petal width, length and sepal width, length from left
data = A.data;

% arrange the data into classes row wise
[samples,~] = size(classes);
class1 = [];
class2 = [];
class3 = [];
k = zeros(1,3);
data_class1 = [];
data_class2 = [];
data_class3 = [];
for i = 1:samples
        if(classes(i,1) == 1)
            class2 = [class2 ; 2];
            data_class2 = [data_class2 ; data(i,:)];
            k(2)=  k(2) + 1;
        elseif(classes(i,2) == 1)
            class3 = [class3 ; 3];
            data_class3 = [data_class3 ; data(i,:)];
            k(3)=  k(3) + 1;
        elseif (classes(i,3) == 1)
            class1 = [class1 ; 1];
            data_class1 = [data_class1 ; data(i,:)];
            k(1)=  k(1) + 1;
        end
end
%%%%%%%%Separating train data and test data and also combining all classes 
%%%%%%%%data together
t_k = (test_percent/100) .* k;
k = (1 -(test_percent/100)) .* k;
class = [ class1(1:k(1),1) ; class2(1:k(2),1) ; class3(1:k(3),1)];
t_class = [class1((k(1)+1):(k(1)+t_k(1)),1) ; class2((k(2)+1):(k(2)+t_k(2)),1) ; class3((k(3)+1):(k(3)+t_k(3)),1)];
data = [data_class1(1:k(1),:) ; data_class2(1:k(2),:) ; data_class3(1:k(3),:)];
t_data = [data_class1((k(1)+1):(k(1)+t_k(1)),:) ; data_class2((k(2)+1):(k(2)+t_k(2)),:) ; data_class3((k(3)+1):(k(3)+t_k(3)),:)];
end