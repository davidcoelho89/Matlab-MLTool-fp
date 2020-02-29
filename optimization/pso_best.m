function [pbest_new,gbest_new] = pso_best(pbest,gbest,X,F,maximization)

% --- Update Best Position (Global and Local) ---
%
%   [pbest_new,gbest_new] = pso_best(pbest,gbest,X,F,maximization)
%
%   Input:
%       pbest = current best local position
%       gbest = current best global position
%       X = current positions
%       F = current fitness
%       maximization = type of problem (maximization or minimization)
%   Output:
%       pbest_new = new best local position
%       gbest_new = new best global position

%% INITIALIZATIONS

[Nc,Ni] = size(X);
pbest_new = zeros(Nc+1,Ni);

%% ALGORITHM

if (maximization == 1), % maximization problem
    for i = 1:Ni,
        if (F(i) > pbest(Nc+1,i)),
            pbest_new(:,i) = [X(:,i) ; F(i)];
        else
            pbest_new(:,i) = pbest(:,i);
        end
    end
    
    [Fbest,ind] = max(F);
    if Fbest > gbest(Nc+1),
        gbest_new = [X(:,ind) ; F(ind)];
    else
        gbest_new = gbest;
    end
else                    % minimization problem
    for i = 1:Ni,
        if (F(i) < pbest(Nc+1,i)),
            pbest_new(:,i) = [X(:,i) ; F(i)];
        else
            pbest_new(:,i) = pbest(:,i);
        end
    end
    
    [Fbest,ind] = min(F);
    if Fbest < gbest(Nc+1),
        gbest_new = [X(:,ind) ; F(ind)];
    else
        gbest_new = gbest;
    end
end

