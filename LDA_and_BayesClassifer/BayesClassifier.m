%% clean up
clc
close all
clear all

%% Training Baye's Classifier for IRIS dataset

f_desiredout = 'Iris Desired.asc';
f_input = 'Iris Input.asc';
%%% Test data is taken to be 40% of the given data
test_percent = 40;
[class, data , k,t_class, t_data , t_k] = arrange_data(f_desiredout,f_input,test_percent);
[sz, ~] = size(data);
% Plotting the parameters of Sepal and  Petals for comparision
figure(1)
gscatter(data(:,3), data(:,4), class,'rgb','osd');
hold on
xlabel('Sepal Length (cm)');
ylabel('Sepal Width (cm)');
legend('Class1','Class2','Class3');
title('Sepal Parameter Variation')
figure(2)
gscatter(data(:,1), data(:,2), class,'rgb','osd');
hold on
xlabel('Petal Length (cm)')
ylabel('Petal Width (cm)')
legend('Class1','Class2','Class3','Location','SouthEast');
title('Petal Parameter Variation')
display('Parameters of Petals seems more efficient for classification than parameters of  Sepal from the plot ');
figure(3)
gscatter(data(:,1), data(:,2), class,'rgb','osd');
hold on
% define variables x1,x2 and x3
syms x1
syms x2
syms x3
[G1 , G2 , G3 ] = bayes_discriminant(x1,x2,x3,data(1:sz/3,1:2),data((sz/3)+1:(2*sz/3),1:2),data((2*sz/3)+1:sz,1:2));
E1 = ezplot(G1 , [0 , 30 , 0 , 70]);
set(E1,'color','r','linestyle','-')
E2= ezplot(G2 , [0 , 30 , 0 , 70]);
set(E2,'color','g','linestyle','-')
E3 = ezplot(G3 , [0 , 30 , 0 , 70]);
set(E3,'color','b','linestyle','-')

title('Bayes Classifier -- IRIS Train DataSet')
xlabel('Petal Length (cm)')
ylabel('Petal Width (cm)')
legend('Class1', 'Class2', 'Class3','Class1|Class2', 'Class2|Class3','Class3|Class1','Location','SouthEast');
hold off

%% Testing the Bayes Classfier to generate Confusion Matrix
[t_sz, ~] = size(t_data);
figure(4)
gscatter(t_data(:,1), t_data(:,2), t_class,'rgb','osd');
hold on
Con_Mat = zeros(3);
count = 1;
val = zeros(t_sz,3);
for j = 1:t_sz/3:t_sz
    for i = j:(t_sz/3)+j-1
    val(i,1) = double(subs(G1,{x1,x2},{t_data(i,1),t_data(i,2)}));
    val(i,2) = double(subs(G2,{x1,x2},{t_data(i,1),t_data(i,2)}));
    val(i,3) = double(subs(G3,{x1,x2},{t_data(i,1),t_data(i,2)}));
        if( val(i,1) > 0 && val(i,2) > 0 && val(i,3) < 0 )
        Con_Mat(1,count) = Con_Mat(1,count) + 1;
        end
        if( val(i,1) < 0 && val(i,2) > 0 && val(i,3) > 0 )
        Con_Mat(2,count) = Con_Mat(2,count) + 1;
        end
        if( val(i,1) < 0 && val(i,2) < 0 && val(i,3) > 0 )
        Con_Mat(3,count) = Con_Mat(3,count) + 1;
        end
    end
count = count + 1;
end

%%%%%Confusion Matrix for Test Data is generated%%%%
False_Predictions = ((sum(t_k)- trace(Con_Mat))*100)/sum(t_k);
display('Confusion Matrix and Percent of False Predictions from IRIS Test Data of 60 samples:');
Con_Mat
False_Predictions
E1 = ezplot(G1 , [0 , 30 , 0 , 70]);
set(E1,'color','r','linestyle','-')
E2= ezplot(G2 , [0 , 30 , 0 , 70]);
set(E2,'color','g','linestyle','-')
E3 = ezplot(G3 , [0 , 30 , 0 , 70]);
set(E3,'color','b','linestyle','-')
title('Bayes Classifier for the Test Data')
xlabel('Petal Length (cm)')
ylabel('Petal Width (cm)')
legend('Class1', 'Class2', 'Class3','Class1|Class2', 'Class2|Class3','Class3|Class1','Location','SouthEast');
hold off