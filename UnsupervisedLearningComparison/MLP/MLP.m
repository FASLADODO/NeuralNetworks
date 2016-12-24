%% clean up
clear all
close all
clc

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
result = [];
mode = 1; %mode 1 for single hidden layer, mode 2 for 3 hidden layers
% mode 3 is for gender data
% run multiple ttimes to see change in accuracy

%% MLP code


switch mode
    case 1
           %  1 hidden layer MLP
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


            inputs = input';
            targets = output';

            hiddenLayerSize = 8;
            net = patternnet(hiddenLayerSize);
            net.trainFcn = 'traingda';

            % Set up Division of Data for Training, Validation, Testing
            net.divideParam.trainRatio = 100/100;
            net.divideParam.valRatio = 0/100;
            net.divideParam.testRatio = 0/100;

            [net,tr] = train(net,inputs,targets);

            % for train data
            outputs = net(inputs);
            errors = gsubtract(targets,outputs);
            performance = perform(net,targets,outputs);

            % for test data
            outputs_test = net(input_test');
            errors_test = gsubtract(output_actual',outputs_test);
            performance_test = perform(net,output_actual',outputs_test);
            net.trainFcn
            net.trainParam.lr
            net.trainParam.lr_inc
            net.trainParam.lr_dec
            cd \\Client\D$\Study\matlab\Homework4\MLP
            con = confusion_matrix(outputs_test',output_actual)

            result = [trace(con)*100/sum(sum(con)) result]
            figure, plotconfusion(output_actual',outputs_test)
      
    case 2
        
        %% 3 hidden layer MLP Species

            
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


            inputs = input';
            targets = output';

            % Create a Pattern Recognition Network
            hiddenLayerSize = [25 20 10];
            net = patternnet(hiddenLayerSize);


            % Set up Division of Data for Training, Validation, Testing
            net.divideParam.trainRatio = 100/100;
            net.divideParam.valRatio = 0/100;
            net.divideParam.testRatio = 0/100;

            [net,tr] = train(net,inputs,targets);

            % for train data
            outputs = net(inputs);
            errors = gsubtract(targets,outputs);
            performance = perform(net,targets,outputs);
            
            % for test data
            outputs_test = net(input_test');
            errors_test = gsubtract(output_actual',outputs_test);
            performance_test = perform(net,output_actual',outputs_test);

            cd \\Client\D$\Study\matlab\Homework4\MLP
            con = confusion_matrix(outputs_test',output_actual);

            result = [trace(con)*100/sum(sum(con)) result];
            figure, plotconfusion(output_actual',outputs_test)
         
        
    case 3
        
        %% 1 hidden layer MLP Gender

        for i=1:3
            
            input = [set2 ; set3];
            input_test = [set1];
            output1 = input(:,7);
            output1_test = input_test(:,7);
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


            inputs = input';
            targets = output';

            % Create a Pattern Recognition Network
            hiddenLayerSize = 10;
            net = patternnet(hiddenLayerSize);


            % Set up Division of Data for Training, Validation, Testing
            net.divideParam.trainRatio = 100/100;
            net.divideParam.valRatio = 0/100;
            net.divideParam.testRatio = 0/100;

            [net,tr] = train(net,inputs,targets);

            % for train data
            outputs = net(inputs);
            errors = gsubtract(targets,outputs);
            performance = perform(net,targets,outputs);
            
            % for test data
            outputs_test = net(input_test');
            errors_test = gsubtract(output_actual',outputs_test);
            performance_test = perform(net,output_actual',outputs_test);

            cd \\Client\D$\Study\matlab\Homework4\MLP
            con = confusion_matrix(outputs_test',output_actual);

            result = [trace(con)*100/sum(sum(con)) result];
            figure, plotconfusion(output_actual',outputs_test)
        end
end