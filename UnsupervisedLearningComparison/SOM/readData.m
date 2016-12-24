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
gender = data(:,7);
inputs = data(:,2:6);
species = data(:,1);

% filename = 'data.txt';
% delimiterIn = '\t';
% headerlinesIn = 0;
% inputs = importdata(filename,delimiterIn,headerlinesIn);
% inputs = inputs';
% species = 0;


%% 1st method: SOM

cluster = 2

switch cluster
    case 2
        batch = 1; % batch training on if 1
        batch_size = 4;
        result = SOM(species,inputs,batch,batch_size);
    case 3
        batch = 1; % batch training on if 1
        batch_size = 2;
        result = SOM(gender,inputs,batch,batch_size);
end
%% 2nd method: MLP 