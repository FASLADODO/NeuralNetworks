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
strarr1 = ['set1' ; 'set2' ; 'set3'];
numMat = [1 2 3];

gender = data(:,7:8);
input = data(:,2:6);
species = data(:,1);
mode = 2; %mode 1 for single hidden layer, mode 2 for 3 hidden layers
% run multiple ttimes to see change in accuracy

%% auto encoder



switch mode
    case 1
        
        %% 1 hidden layer Autoencoder

            
            input = [set2 ; set3];
            input_test = [set1];
            output1 = input(:,1);
            output1_test = input_test(:,1);
            input = input(:,2:6);
            input_test = input_test(:,2:6);
            [sz,~] = size(input);
            [sz1,~] = size(input_test);
            
            output = [];
            output_actual = [];
            for k = 1:sz
                a1 = output1(k,1);
                if(a1 == 1)
                    b1 = 0;
                    c1 = 1;
                else
                    b1 = 1;
                    c1 = 0;
                end
                output = [output; [c1 b1]];
            end
            for k = 1:sz1
                a1 = output1_test(k,1);
                if(a1 == 1)
                    b1 = 0;
                    c1 = 1;
                else
                    b1 = 1;
                    c1 = 0;
                end
                output_actual = [output_actual; [c1 b1]];
            end


            X = input';
            tTrain = output';
        
            % highest achieved 84%
            autoenc = trainAutoencoder(X,2,'MaxEpochs',1000,'EncoderTransferFunction','logsig',...
            'DecoderTransferFunction','logsig','L2WeightRegularization',0.01,...
        'SparsityRegularization',5,...
        'SparsityProportion',0.15);
            feat = encode(autoenc,X);
            XReconstructed = predict(autoenc,X);
            accuracy = (1- mse(X-XReconstructed))*100
            
            testReconstructed = predict(autoenc,input_test');
            accuracy = (1- mse(input_test'-testReconstructed))*100
            figure;
            plot(input_test','r.');
            hold on
            plot(testReconstructed,'go');
            hold off
            
            softnet = trainSoftmaxLayer(feat,tTrain,'MaxEpochs',500);
            deepnet = stack(autoenc,softnet);
            deepnet = train(deepnet,X,tTrain);
            y = deepnet(X);
            figure,
            plotconfusion(tTrain,y);
            figure,
            testy = deepnet(input_test');
            plotconfusion(output_actual',testy);
            % cluster using k-means
            feat = feat';
            [idx,C,sumd,D] = kmeans(feat,2);

            figure;
            plot(feat(idx==1,1),feat(idx==1,2),'r.','MarkerSize',12)
            hold on
            plot(feat(idx==2,1),feat(idx==2,2),'b.','MarkerSize',12)
            plot(C(:,1),C(:,2),'kx',...
                 'MarkerSize',15,'LineWidth',3)
             
            legend('Cluster 1','Cluster 2','Centroids',...
                   'Location','NW')
            title 'Species Cluster Assignments and Centroids'
            hold off

    case 2

           %% 3 hidden layer auto encoder
           
            
            input = [set2 ; set3];
            input_test = [set1];
            output1 = input(:,1);
            output1_test = input_test(:,1);
            input = input(:,2:6);
            input_test = input_test(:,2:6);
            [sz,~] = size(input);
            [sz1,~] = size(input_test);
            
            output = [];
            output_actual = [];
            for k = 1:sz
                a1 = output1(k,1);
                if(a1 == 1)
                    b1 = 0;
                    c1 = 1;
                else
                    b1 = 1;
                    c1 = 0;
                end
                output = [output; [c1 b1]];
            end
            for k = 1:sz1
                a1 = output1_test(k,1);
                if(a1 == 1)
                    b1 = 0;
                    c1 = 1;
                else
                    b1 = 1;
                    c1 = 0;
                end
                output_actual = [output_actual; [c1 b1]];
            end


            X = input';
            tTrain = output';
        rng('default')

        hiddenSize1 = 4;
        autoenc1 = trainAutoencoder(X,hiddenSize1,'MaxEpochs',500,'EncoderTransferFunction','logsig',...
        'DecoderTransferFunction','logsig', 'L2WeightRegularization',0.004, ...
        'SparsityRegularization',4, 'SparsityProportion',0.15);
        feat1 = encode(autoenc1,X);

        hiddenSize2 = 3;
        autoenc2 = trainAutoencoder(feat1,hiddenSize2,'MaxEpochs',500,'EncoderTransferFunction','logsig',...
        'DecoderTransferFunction','logsig', 'L2WeightRegularization',0.004, ...
        'SparsityRegularization',4, 'SparsityProportion',0.15);
        feat2 = encode(autoenc2,feat1);

        hiddenSize3 = 2;
        autoenc3 = trainAutoencoder(feat2,hiddenSize3,'MaxEpochs',500,'EncoderTransferFunction','logsig',...
        'DecoderTransferFunction','logsig', 'L2WeightRegularization',0.004, ...
        'SparsityRegularization',4, 'SparsityProportion',0.15);
        feat3 = encode(autoenc3,feat2);

        softnet = trainSoftmaxLayer(feat3,tTrain,'MaxEpochs',500);

        deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);
        deepnet = train(deepnet,X,tTrain);

        y = deepnet(X);
        plotconfusion(tTrain,y);
        figure,
        testy = deepnet(input_test');
        plotconfusion(output_actual',testy);
                % cluster using k-means
        feat = feat3';
        [idx,C,sumd,D] = kmeans(feat,2);

        figure;
        plot(feat(idx==1,1),feat(idx==1,2),'r.','MarkerSize',12)
        hold on
        plot(feat(idx==2,1),feat(idx==2,2),'b.','MarkerSize',12)
        plot(C(:,1),C(:,2),'kx',...
             'MarkerSize',15,'LineWidth',3)
        legend('Cluster 1','Cluster 2','Centroids',...
               'Location','NW')
        title 'Species Cluster Assignments and Centroids'
        hold off

end