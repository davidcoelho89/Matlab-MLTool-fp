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

%% Animacao pendulo

dt = 0.01;
[tout,x] = ode45(@pendulo,0:dt:20,[pi/4;0]);

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
	plot([0 vetx(n)],[0 vety(n)],'b-', ...
          vetx(n),vety(n),'bo', ...
          [-1 1],[0 0],'b-');
	axis([-1.5 1.5 -1.5 1.5]);
	subplot(1,2,2);
	hold on
	plot(x(n,1),x(n,2),'r.');
	hold off;
	pause(dt);
end

%% END