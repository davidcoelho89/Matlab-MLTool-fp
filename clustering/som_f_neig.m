function [h] = som_f_neig (neig,r_k,r_win,Nn,t,tmax,Vo,Vt)

% --- SOM Neighborhood Function ---
%
%   [h] = som_f_neig (neig,r_k,r_win,Nn,t,tmax,Vo,Vt)
%
%   Input:
%       neig = type of neighborhood function
%           1:  if winner or neighbor, h = 1, else h = 0.
%           ~1: if neighbor, h = exp (-(||ri - ri*||^2)/(V^2))
%                where: V = Vo*((Vt/Vo)^(t/tmax))
%       r_k = current prototypes position
%       r_win = winner prototype position
%       Nn = number of neighbors
%       t = current iteration
%       tmax = max number of iterations
%       Vo = initial value of V
%       Vt = final value of V
%   Output:
%       h = neigborhood function result

%% ALGORITHM

if neig == 1
    if max(abs(r_k - r_win)) > Nn
        h = 0;
    else
        h = 1;
    end
else
    if max(abs(r_k - r_win)) > Nn
        h = 0;
    else
        V = Vo*((Vt/Vo)^(t/tmax));
        h = exp(-sum((r_win - r_k).^2)/(V^2));
    end
end
    
%% END