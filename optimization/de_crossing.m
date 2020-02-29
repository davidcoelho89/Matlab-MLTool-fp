function Unew = de_crossing(X,U,pc)

% selecao de indiduos pelo metodo do sorteio
%
%   Unew = de_crossing(X,U,pc)
%
%   Entradas:
%       X - Current Population              [Nc x Ni]
%       U - Trial Vectors                   [Nc x Ni]
%       pc - crossing probability           [cte]
%
%   Saidas:
%       Unew - New trial Vector             [Nc x Ni]

%% INITIALIZATIONS

[Nc,Ni] = size(X);
Unew = zeros(Nc,Ni);

%% ALGORITHM

for i = 1:Ni,
    for j = 1:Nc,
        % Aleatory generates number between 0 - 1
        r = rand(1);
        % Generate new trial vectors
        if r < pc,
            Unew(j,i) = U(j,i);
        else
            Unew(j,i) = X(j,i);
        end
    end
end

%% FILL OUTPUT STRUCTURE



%% END