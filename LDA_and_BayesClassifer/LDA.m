%% clean up
clc
close all
clear all

%% Linear Discriminant classifier

f_desiredout = 'Iris Desired.asc';
f_input = 'Iris Input.asc';
%%% Test data is taken to be 40% of the given data
test_percent = 40;
[classes, data , k,t_class, t_data , t_k] = arrange_data(f_desiredout,f_input,test_percent);
%%%%% Plot data %%%%%%%%%%%%%%%%
figure(1)
plot ( data(1:k(1),2),data(1:k(1),1),'+');
hold on
plot ( data((k(1)+1):(k(1)+k(2)),2),data((k(1)+1):(k(1)+k(2)),1),'*');
plot ( data((k(1)+k(2)+1):(k(1)+k(2)+k(3)),2),data((k(1)+k(2)+1):(k(1)+k(2)+k(3)),1),'^');
xlabel('Petal Length (cm)') % x-axis label
ylabel('Petal Width (cm)') % y-axis label
legend('Class1', 'Class2', 'Class3' );
title('Petal Parameter Variation')
grid on;
hold off;
figure(2)
plot ( data(1:k(1),4),data(1:k(1),3),'+');
hold on
plot ( data((k(1)+1):(k(1)+k(2)),4),data((k(1)+1):(k(1)+k(2)),3),'*');
plot ( data((k(1)+k(2)+1):(k(1)+k(2)+k(3)),4),data((k(1)+k(2)+1):(k(1)+k(2)+k(3)),3),'^');
xlabel('Sepal Length (cm)') % x-axis label
ylabel('Sepal Width (cm)') % y-axis label
legend('Class1', 'Class2', 'Class3' );
title('Sepal Parameter Variation')
grid on;
hold off;
%%%%% Get the means and covariances of each type of data for each class
[mean_matrix] = get_mean(classes,data,k);
overall_mean = mean(data,1);
[sum_covar, covar] = within_class_scatter(classes,data,k,mean_matrix);
%%%%Transforming two dimensional LDA eigen space
[eigv,eigvec, ld,tranformdata]=lda_space(data,sum_covar,mean_matrix,k,overall_mean);
%%%%%%%%%%%%%%Plot the transformed data in LDA space%%%%%%%%%%
figure(3)
plot ( tranformdata(1:k(1),1),tranformdata(1:k(1),2),'+');
hold on
plot ( tranformdata((k(1)+1):(k(1)+k(2)),1),tranformdata((k(1)+1):(k(1)+k(2)),2),'*');
plot ( tranformdata((k(1)+k(2)+1):(k(1)+k(2)+k(3)),1),tranformdata((k(1)+k(2)+1):(k(1)+k(2)+k(3)),2),'^');
syms x1 x2
X = [x1 x2];
%%%%%%%%%%%Getting the decision boundary equations along with weights and
%%%%%%%%%%%constants 
[ G1, G2, G3, o, w, c] = ldaDiscriminant(classes, tranformdata, k, X);
%%%% Plotting the decision boundaries %%%%%%%%
E1 = ezplot(G1 , [-45 , 15 , -35 , -15]);
set(E1,'color','r','linestyle','-')
E2 = ezplot(G2 , [-45 , 15 , -35 , -15]);
set(E2,'color','g','linestyle','-')
E3 = ezplot(G3 , [-45 , 15 , -35 , -15]);
set(E3,'color','b','linestyle','-')
xlabel('LD1') % x-axis label
ylabel('LD2') % y-axis label
legend('Class1', 'Class2', 'Class 3','Class1|Class2', 'Class2|Class3','Class3|Class1');
title('Linear Discriminant Classifier -- IRIS Train DataSet')
grid on;
hold off;

%% Testing the LDA Classfier to generate Confusion Matrix
t_tranformdata = t_data*(eigvec(1:ld,:))';
[Con_Mat] = ConfusionMatrix( t_tranformdata,w,c,t_class,t_k);
False_Predictions = ((sum(t_k)- trace(Con_Mat))*100)/sum(t_k);
display('Confusion Matrix and Percent of False Predictions from IRIS Test Data of 60 samples:');
Con_Mat
False_Predictions 
figure(4)
plot ( t_tranformdata(1:t_k(1),1),t_tranformdata(1:t_k(1),2),'+');
hold on
plot ( t_tranformdata((t_k(1)+1):(t_k(1)+t_k(2)),1),t_tranformdata((t_k(1)+1):(t_k(1)+t_k(2)),2),'*');
plot ( t_tranformdata((t_k(1)+t_k(2)+1):(t_k(1)+t_k(2)+t_k(3)),1),t_tranformdata((t_k(1)+t_k(2)+1):(t_k(1)+t_k(2)+t_k(3)),2),'^');
E1 = ezplot(G1 , [-50 , 15 , -30 , -15]);
set(E1,'color','r','linestyle','-')
E2 = ezplot(G2 , [-50 , 15 , -30 , -15]);
set(E2,'color','g','linestyle','-')
E1 = ezplot(G3 , [-50 , 15 , -30 , -15]);
set(E3,'color','b','linestyle','-')
xlabel('LD1') % x-axis label
ylabel('LD2') % y-axis label
legend('Class1', 'Class2', 'Class3','Class1|Class2', 'Class2|Class3','Class3|Class1');
title('Linear Discriminant Classifier -- IRIS Test DataSet')
grid on;
hold off;

