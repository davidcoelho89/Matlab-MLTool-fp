function [n] = prototypes_learn(learn,tmax,t,No,Nt)

% --- SOM 1-D Learning function ---
%
%   [n] = som1d_f_learn(learn,tmax,t,No,Nt);
%
%   Input:
%       learn = type of learning function
%           1: N = No (constant)
%           2: N = No*(1-(t/tmax))
%           3: N = No/(1+t)
%           4: N = No*((Nt/No)^(t/tmax))
%       tmax = max number of iterations
%       t = current iteration
%       No = initial learning step
%       Nt = final learning step
%   Output:
%       n = learning function result

%% ALGORITHM

if learn == 1,
    n = No;
elseif learn == 2,
    n = No*(1-(t/tmax));
elseif learn == 3,
    n = No/(1+t);
elseif learn == 4,
    n = No*((Nt/No)^(t/tmax));
end

%% END