function [Vnew] = pso_vel(X,V,pbest,gbest,W,c1,c2,r1,r2)

% --- Update Velocity of Particles
% 
%   [Vnew] = pso_vel(V,X,pbest,gbest,W,c1,c2,r1,r2)
%
%   Input:
%       X = Previous Positions                          [Nc x Ni]
%       V = Previous Velocity                           [Nc x Ni]
%       pbest = best local position                     [Nc x Ni]
%       gbest = best overall position                   [Nc x 1]
%       W = Inertia Coefficient                         [Nc x 1]
%       c1 = importance of the best local value         [Nc x 1]
%       c2 = importance of the best general value       [Nc x 1]
%   Output:
%       Vnew = new velocity                             [Nc x 1]

%% INITIALIZATIONS

[Nc,Ni] = size(V);
Vnew = zeros(Nc,Ni);

%% ALGORITHM

for i = 1:Ni,
    for c = 1:Nc,
       Vnew(c,i) = W*V(c,i) + c1*r1(c,i)*( pbest(c,i) - X(c,i) ) ...
                   + c2*r2(c,i)*( gbest(c) - X(c,i) );
    end
end

%% FILL OUTPUT STRUCTURE



%% END