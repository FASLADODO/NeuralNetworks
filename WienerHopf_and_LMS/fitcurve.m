%1d curve fitting for the equation provided

function fitcurve(w, input , output)

[samples, ~] = size(output);

x = linspace(min(input(:,1)),max(input(:,1)), samples);
y = (w(1).*x) + w(2) ;

figure
plot(x,y)
hold on;
plot(input,output,'+')
title('Curve fitted to desired points')
xlabel('Input') 
ylabel('output') 
legend('Predicted curve','Actual points')
axis([-4 4 0 100])
grid on;
hold off;

end