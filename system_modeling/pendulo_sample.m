%% Pendulo - Teste

clear;
clc;
close;

%% Grafico das variaves e de fase

dt = 0.01;
[tout,x] = ode45(@pendulo,0:dt:20,[pi/4;0]);

figure;
grid
plot(tout,x(:,1),tout,x(:,2))

figure;
grid
plot(x(:,1),x(:,2))

%% Animacao Pendulo Suspenso

dt = 0.01;
[~,x] = ode45(@pendulo,0:dt:20,[pi/4;0]);

l = 1;

N = size(x,1);
vetx = zeros(1,N);
vety = zeros(1,N);

for n = 1:N
	vetx(n) = l*cos(x(n,1)-pi/2);
	vety(n) = l*sin(x(n,1)-pi/2);
end

figure;
subplot(1,2,2);
plot(x(:,1),x(:,2));
for n = 1:N
	subplot(1,2,1);
	plot([0 vetx(n)],[0 vety(n)],'b-', ...      % Reta do pendulo
          vetx(n),vety(n),'bo', ...             % Extremo do pendulo
          [-1 1],[0 0],'b-'...                  % Referencia
         );
	axis([-1.5 1.5 -1.5 1.5]);
	subplot(1,2,2);
	hold on
	plot(x(n,1),x(n,2),'r.');
	hold off;
	pause(dt);
end

%% Grafico das variaves

dt = 0.01;
[tout,x] = ode45(@pendulo_invertido,0:dt:60,[0;0;8*pi/9;0]);

figure;
grid
plot(tout,x(:,1),tout,x(:,3))
legend({'x','theta'},'Location','northwest')

%% Animacao Pendulo Invertido

dt = 0.01;
[tout,x] = ode45(@pendulo_invertido,0:dt:40,[0;0;10*pi/9;0]);

pos = x(:,1);
angle = x(:,3);

l = 0.3;    % comprimento do pendulo
hc = 0.15;	% altura do carro
lc = 0.2;   % largura carro

N = length(pos);
vetx = zeros(1,N);
vety = zeros(1,N);

for n = 1:N
	vetx(n) = pos(n) + l*cos(angle(n)-pi/2);
	vety(n) = hc + l*sin(angle(n)-pi/2);
end

figure;
for n = 1:N
	plot([pos(n) vetx(n)],[hc vety(n)],'b-', ...      % Reta do pendulo
          vetx(n),vety(n),'bo', ...                   % Extremo do pendulo
          [-1 1],[0 0],'b-', ...                      % Referencia
          [pos(n)-lc/2,pos(n)+lc/2],[hc hc],'b-', ...     % Topo carro
          [pos(n)-lc/2,pos(n)+lc/2],[hc/2 hc/2],'b-', ... % Base carro
          [pos(n)-lc/2,pos(n)-lc/2],[hc/2 hc],'b-', ...   % Esq. carro
          [pos(n)+lc/2,pos(n)+lc/2],[hc/2 hc],'b-', ...   % Dir. carro
          pos(n)-lc/2,hc/4,'bo', ...                  % Roda Esq.
          pos(n)+lc/2,hc/4,'bo' ...                   % Roda Dir
         );
	axis([-1.5 1.5 -1.5 1.5]);
	pause(dt);
end

%% END