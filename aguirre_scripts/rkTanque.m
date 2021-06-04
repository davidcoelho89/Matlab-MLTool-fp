function x = rkTanque(x0,u,h,t)
%
% x = rkTanque(x0,u,h,t)
% 
% Implementa o algoritmo de integracao numerica Runge Kuta de 4a ordem
% x0: vetor de estado (antes de chamar a funcao... eh a condicao inicial)
% u: se houver, eh o valor de entrada, considerado constante ao longo do
%    intervalo de integracao, h
% h: intervalo de integracao
% t: instante de tempo logo antes de chamar a funcao de integracao
% O campo vetorial deve ser escrito no corpo da funcao dvNOME

% LAA 17/03/2017

% 1a chamada
xd = dvTanque(x0,u,t);
savex0 = x0;
phi = xd;
x0 = savex0 + 0.5*h*xd;

% 2a chamada
xd = dvTanque(x0,u,t+0.5*h);
phi = phi + 2*xd;
x0 = savex0 + 0.5*h*xd;

% 3a chamada
xd = dvTanque(x0,u,t+0.5*h);
phi = phi + 2*xd;
x0 = savex0 + h*xd;

% 4a chamada
xd = dvTanque(x0,u,t+h);
x = savex0 + (phi+xd)*h/6;
