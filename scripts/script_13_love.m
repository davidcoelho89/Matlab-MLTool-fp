% L (y = 1/x)
figure;
x = 0.01:0.01:1;
y = 1./x;
plot(x,y);

% O (x^2+y^2 = 9)
figure;
x1 = -3:0.01:3;
y1 = (9-x1.^2).^0.5;
x2 = -3:0.01:3;
y2 = -(9-x2.^2).^0.5;
plot(x1,y1);
hold on
plot(x2,y2);
hold off

% V (y = |-2x|)
figure;
x = -1:0.01:1;
y = abs(-2*x);
plot(x,y);

% E (x = -3|sin y|)
figure;
y = -3.14:0.01:3.14;
x = -3*abs(sin(y));
plot(x,y);
axis([-5 2 -3.2 3.2])
