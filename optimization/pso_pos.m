function [Xnew] = pso_pos(X,V)

% --- Update Position of Particles --
%
%   [Xnew] = pso_pos(X,V)
%
%   Input:
%       X = Previous Positions                          [Nc x Ni]
%       V = Previous Velocity                           [Nc x Ni]
%   Output:
%       Xnew = new Position                             [Nc x 1]


%% INITIALIZATIONS



%% ALGORITHM

Xnew = X + V;

%% FILL OUTPUT STRUCTURE



%% END