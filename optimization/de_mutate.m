function [V] = de_mutate(X,B)

% --- Generate Trial Vectors for DE ---
%
%   Pout = ga_mutate(P,Pm,Nc,El)
%
%   Input:
%       X = population of subjects       	[Nc x Ni]
%       B = Difference anplification        [cte]
%
%   Output:
%       V = trial vectors                   [Nc x Ni]

%% INITIALIZATIONS

[Nc,Ni] = size(X);
V = zeros(Nc,Ni);

%% ALGORITHM

for i = 1:Ni,
    
    % Generate vector [1 x Ni] with random sequence
    R = randperm(Ni);
    
    % Ensure that the three first vectors are diferent from X(:,i)
    ind = find(R == i);
    if ind < 4,
        R(ind) = R(4);
    end
    
    r1 = R(1);	% target vector
    r2 = R(2);	% first individual
    r3 = R(3); 	% second individual
    
    for j = 1:Nc,
        V(j,i) = X(j,r1) + B*(X(j,r2) - X(j,r3));
    end
    
end

%% FILL OUTPUT STRUCTURE



%% END