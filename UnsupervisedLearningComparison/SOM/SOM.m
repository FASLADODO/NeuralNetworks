% SOM function

function result = SOM(species,inputs,batch,batch_size)

%% online training

if batch == 0
    % unit1 = [.2 .6 .5 .9 .3];
    % unit2 = [.8 .4 .7 .3 .2];
    unit1 = [0.5 0.5 0.5 0.5 0.5];
    unit2 = [0.5 0.5 0.5 0.5 0.5];
    lr = 0.5;
    epoch = 200;
    u11 = unit1(1); u12 = unit1(2); u13 = unit1(3); u14 = unit1(4); u15 = unit1(5);
    u21 = unit2(1); u22 = unit2(2); u23 = unit2(3); u24 = unit2(4); u25 = unit2(5);

    [sz,~] = size(inputs);

    for j = 1:epoch
        for i = 1:sz
            d1 = (u11-inputs(i,1))^2 + (u12-inputs(i,2))^2 +...
                (u13-inputs(i,3))^2 + (u14-inputs(i,4))^2 + (u15-inputs(i,5))^2;
            d2 = (u21-inputs(i,1))^2 + (u22-inputs(i,2))^2 +...
                (u23-inputs(i,3))^2 + (u24-inputs(i,4))^2 + (u25-inputs(i,5))^2;
            if d1 < d2
               u11 = u11 + lr * ( inputs(i,1) - u11); 
               u12 = u12 + lr * ( inputs(i,2) - u12); 
               u13 = u13 + lr * ( inputs(i,3) - u13); 
               u14 = u14 + lr * ( inputs(i,4) - u14); 
               u15 = u15 + lr * ( inputs(i,5) - u15); 
            else
               u21 = u21 + lr * ( inputs(i,1) - u21); 
               u22 = u22 + lr * ( inputs(i,2) - u22); 
               u23 = u23 + lr * ( inputs(i,3) - u23); 
               u24 = u24 + lr * ( inputs(i,4) - u24); 
               u25 = u25 + lr * ( inputs(i,5) - u25);
            end
        end
    end

    unit1 = [u11 u12 u13 u14 u15];
    unit2 = [u21 u22 u23 u24 u25];

    output = [];
    for i = 1:sz
        d1 = (u11-inputs(i,1))^2 + (u12-inputs(i,2))^2 +...
            (u13-inputs(i,3))^2 + (u14-inputs(i,4))^2 + (u15-inputs(i,5))^2;
        d2 = (u21-inputs(i,1))^2 + (u22-inputs(i,2))^2 +...
            (u23-inputs(i,3))^2 + (u24-inputs(i,4))^2 + (u25-inputs(i,5))^2;
        if d1 < d2
            output = [output 0];
        else
            output = [output 1];
        end
    end
    output = output';

    output_actual = [];
    output_desired = [];
    for k = 1:sz
        a1 = output(k,1);
        a2 = species(k,1);
        if(a1 == 1)
            b1 = 0;
            c1 = 1;
        else
            b1 = 1;
            c1 = 0;
        end
        output_actual = [output_actual; [c1 b1]];
        if(a2 == 0)
            b1 = 0;
            c1 = 1;
        else
            b1 = 1;
            c1 = 0;
        end
        output_desired = [output_desired; [c1 b1]];
    end

    con = confusion_matrix(output_desired,output_actual)
    result = trace(con)*100/sum(sum(con))

%% batch training

else
    
        % unit1 = [.2 .6 .5 .9 .3];
    % unit2 = [.8 .4 .7 .3 .2];
    unit1 = [10.4532 9.3346 21.9579 25.3445 9.2256];
    unit2 = [17.1977 14.7858 34.9533 39.4757 15.1827];
    lr = 0.5;
    epoch = 500;
    u11 = unit1(1); u12 = unit1(2); u13 = unit1(3); u14 = unit1(4); u15 = unit1(5);
    u21 = unit2(1); u22 = unit2(2); u23 = unit2(3); u24 = unit2(4); u25 = unit2(5);

    [sz,~] = size(inputs);
%     inputs1 = sum(inputs)/200;

    for j = 1:epoch
        for i = 1:batch_size:sz
            if batch_size ~= 1
                inputs1 = sum(inputs(i:i+batch_size-1,:))/(batch_size);
            else
                inputs1 = (inputs(i:i+batch_size-1,:))/(batch_size);
            end

            d1 = (u11-inputs1(:,1))^2 + (u12-inputs1(:,2))^2 +...
                (u13-inputs1(:,3))^2 + (u14-inputs1(:,4))^2 + (u15-inputs1(:,5))^2;
            d2 = (u21-inputs1(:,1))^2 + (u22-inputs1(:,2))^2 +...
                (u23-inputs1(:,3))^2 + (u24-inputs1(:,4))^2 + (u25-inputs1(:,5))^2;
            if d1 < d2
                u11 = u11 + lr * ( inputs1(:,1) - u11);
                u12 = u12 + lr * ( inputs1(:,2) - u12);
                u13 = u13 + lr * ( inputs1(:,3) - u13);
                u14 = u14 + lr * ( inputs1(:,4) - u14);
                u15 = u15 + lr * ( inputs1(:,5) - u15);
            else
                u21 = u21 + lr * ( inputs1(:,1) - u21);
                u22 = u22 + lr * ( inputs1(:,2) - u22);
                u23 = u23 + lr * ( inputs1(:,3) - u23);
                u24 = u24 + lr * ( inputs1(:,4) - u24);
                u25 = u25 + lr * ( inputs1(:,5) - u25);
            end
        
        end
    end

    unit1 = [u11 u12 u13 u14 u15];
    unit2 = [u21 u22 u23 u24 u25];

    output = [];
    for i = 1:sz
        d1 = (u11-inputs(i,1))^2 + (u12-inputs(i,2))^2 +...
            (u13-inputs(i,3))^2 + (u14-inputs(i,4))^2 + (u15-inputs(i,5))^2;
        d2 = (u21-inputs(i,1))^2 + (u22-inputs(i,2))^2 +...
            (u23-inputs(i,3))^2 + (u24-inputs(i,4))^2 + (u25-inputs(i,5))^2;
        if d1 < d2
            output = [output 0];
        else
            output = [output 1];
        end
    end
    output = output';

    output_actual = [];
    output_desired = [];
    for k = 1:sz
        a1 = output(k,1);
        a2 = species(k,1);
        if(a1 == 1)
            b1 = 0;
            c1 = 1;
        else
            b1 = 1;
            c1 = 0;
        end
        output_actual = [output_actual; [c1 b1]];
        if(a2 == 0)
            b1 = 0;
            c1 = 1;
        else
            b1 = 1;
            c1 = 0;
        end
        output_desired = [output_desired; [c1 b1]];
    end

    con = confusion_matrix(output_desired,output_actual)
    result = trace(con)*100/sum(sum(con))
    
end
end