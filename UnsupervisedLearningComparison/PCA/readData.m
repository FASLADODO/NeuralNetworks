%% close all
clc
close all
clear all

%% read data

% load data
filename = 'crab_full.txt';
delimiterIn = '\t';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);
data = A.data;

[sz1,~] = size(data);
sz = round(sz1/3);
set1 = data(1:sz , :);
set2 = data((sz+1):(sz*2), :);
set3 = data(((sz*2)+1):sz1 , :);

%% 1st method: PCA


    clusters = 2;  
    input = [set2; set3];
    input_test = set1 ;
    output1 = input(:,1);
    output1_test = input_test(:,1);
    input = input(:,2:6);
    input_test = input_test(:,2:6);
    [sz,~] = size(input);
    [sz1,~] = size(input_test);
    
    result = PCA(output1,input,output1_test,input_test,clusters);
