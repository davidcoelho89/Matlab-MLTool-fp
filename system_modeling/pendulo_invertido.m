function dx = pendulo_invertido(t,x)

% Inverted pendulum
%
% M = 0.5 ; m = 0.2 ; b = 0.1 ; l = 0.3 ; I = 0.006 ; g = 10 ;
%
% "Em relação ao eixo linear:"
% "x1 = estado 1 é o deslocamento linear"
% "x2 = estado 2 é a velocidade linear"
%
% "Em relação ao eixo de rotacao":
% "x3 = estado 3 é o angulo"
% "x4 = estado 4 é a velocidade angular"

%% Parametros do sistema

F = 0;
m = 0.2;
M = 0.5;
b = 0.1;
l = 0.3;
g = 10;
% I = 0.006;

%% Modelagem 3

s3 = sin(x(3));
c3 = cos(x(3));
den = M + m*(1 - c3^2);

dx(1,1) = x(2);

dx(2,1) = F - b*x(2) + m*g*s3*c3 - m*l*(x(4)^2)*s3 / den;

dx(3,1) = x(4);

dx(4,1) = (F - b*x(2))*c3 - m*l*(x(4)^2)*s3*c3 + g*(M+m)*s3 / (den*l);

%% Modelagem 1

% s3 = sin(x(3));
% c3 = cos(x(3));
% t3 = tan(x(3));
% 
% dx(1,1) = x(2);
% 
% dx(2,1) = -(F + (m*l*s3)*x(4) - b*x(2) + g*(m*(l^2))*s3*c3)* ...
%            (I + m*(l^2))/( (m*l*c3)^2 - (M+m)*(I+m*(l^2)) );
% 
% dx(3,1) = x(4);
% 
% dx(4,1) = (F + (m*l*s3)*x(4) - b*x(2) + (M+m)*g*t3 ) * ...
%            (m*l*c3)/( (m*l*c3)^2 - (M+m)*(I+m*(l^2)) );

%% Modelagem 2

% dx(1,1) = x(2);

% dx(2,1) = -(F + (m*l*sin(x(3)))*x(4) + g*(m*(l^2))*sin(x(3))*cos(x(3)))* ...
%           (I + m*(l^2))/( (m*l*cos(x(3)))^2 - (M+m)*(I+m*(l^2)) );

% dx(3,1) = x(4);

% dx(4,1) = (F + (m*l*sin(x(3)))*x(4) + (M+m)*g*tan(x(3)) ) * ...
%            (m*l*cos(x(3)))/( (m*l*cos(x(3)))^2 - (M+m)*(I+m*(l^2)) );

%% End
