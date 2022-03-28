function dx = pendulo(t,x)

% Suspended pendulum
%
% m.l.d2theta/dt^2 = -m.g.sin(theta) -k.l.theta
% d2theta/dt^2 = -(g/l).sin(theta) - (k/m).theta
%
% k = 1 ; g = 10 ; m = 1 ; l = 1;
% atrito de fricção ; gravidade ; massa ; comprimento
%
% "em relação ao eixo vertical:
% "estado 1 é o angulo"
% "estado 2 é a velocidade angular"

dx(1,1) = x(2);
dx(2,1) = -10*sin(x(1)) - x(2);