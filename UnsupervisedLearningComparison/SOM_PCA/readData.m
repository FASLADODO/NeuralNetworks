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
gender = data(:,7:8);
input = data(:,2:6);
species = data(:,1);

%% 1st method:  PCA with SOM clustering

clusters = 2;
% for species
% accuracy of 91 for batch = 1 and batch_size = 10
% accuracy of 87.5 for batch = 1 and batch_size = 5
% accuracy of 85 for batch = 0
% accuracy of 85 for gui = 1
% for gender
% accuracy of 91.5% for batch = 1, batch_size = 5
% accuracy of 87.5 for batch = 1 and batch_size = 10
% accuracy of 52.5 for batch = 0
% accuracy of 49 for gui = 1
batch = 1; % for batch SOM batch = 1
batch_size = 10;
gui = 0;
result = PCA_SOM(species,gender(:,1),input,clusters, batch, batch_size , gui);

%% 2nd method: MLP 