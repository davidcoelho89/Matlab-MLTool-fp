function Xnew = de_select(X,U,Fx,Fu)

% --- Generates New Population --
%
%   Xnew = de_select(X,U,F1,F2)
%
%   Input:
%       X = population of subjects       	[Nc x Ni]
%       U = trial vectors                   [Nc x Ni]
%       Fx = Fitness of population          [1 x Ni]
%       Fu = Fitness of trial vectors       [1 x Ni]
%
%   Output:
%   Xnew = New generation                   [Nc,Ni]

%% INITIALIZATIONS

[Nc,Ni] = size(X);
Xnew = zeros(Nc,Ni);

%% ALGORITHM

for i = 1:Ni,
    if Fx(i) < Fu(i),
        Xnew(:,i) = X(:,i);
    else
        Xnew(:,i) = U(:,i);
    end
end

%% FILL OUTPUT STRUCTURE



%% END