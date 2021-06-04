function xdot = dvTanque(x,u,t)
%
% xdot = dvTanque(x,u,t)
%
% implementa a equacao diferencial de um tanque
% x: nivel => unica variavel de estado
% u: vazao de entrada (qe)
% xd derivada de x (valor do campo vetorial em x)

% LAA 17/03/2017

% valores hipoteticos dos parametros
C = 1; % area constante do tanque
K = 0.5; % constante do registro

% Differential equations
qs = K*sqrt(x);
xd = (u-qs)/C;

% Para evitar erros numericos (o nivel nao pode ser negativo!)
if abs(x) < 0.01
   xd = 0.01;
end

xdot = xd';