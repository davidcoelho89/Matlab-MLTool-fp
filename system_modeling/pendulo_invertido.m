function dx = pendulo_invertido(t,x)

% Inverted pendulum
%
% M = 0.5 ; m = 0.2 ; b = 0.1 ; l = 0.3 ; I = 0.006 ; g = 10 ;
%
% "em relação ao eixo linear:"
% "estado 1 é o deslocamento linear"
% "estado 2 é a velocidade linear"
% "em relação ao eixo vertical":
% "estado 3 é o angulo"
% "estado 4 é a velocidade angular"

F = 0;
m = 0.2;
M = 0.5;
b = 0.1;
l = 0.3;
I = 0.006;
g = 10;

dx(1,1) = x(2);

dx(2,1) = -(F + (m*l*sin(x(3)))*x(4) - b*x(2) + g*(m*(l^2))*sin(x(3))*cos(x(3)))* ...
          (I + m*(l^2))/( (m*l*cos(x(3)))^2 - (M+m)*(I+m*(l^2)) );

% dx(2,1) = -(F + (m*l*sin(x(3)))*x(4) + g*(m*(l^2))*sin(x(3))*cos(x(3)))* ...
%           (I + m*(l^2))/( (m*l*cos(x(3)))^2 - (M+m)*(I+m*(l^2)) );

dx(3,1) = x(4);

dx(4,1) = (F + (m*l*sin(x(3)))*x(4) - b*x(2) + (M+m)*g*tan(x(3)) ) * ...
           (m*l*cos(x(3)))/( (m*l*cos(x(3)))^2 - (M+m)*(I+m*(l^2)) );

% dx(4,1) = (F + (m*l*sin(x(3)))*x(4) + (M+m)*g*tan(x(3)) ) * ...
%            (m*l*cos(x(3)))/( (m*l*cos(x(3)))^2 - (M+m)*(I+m*(l^2)) );
