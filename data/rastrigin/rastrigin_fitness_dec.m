function [OUT] = rastrigin_fitness_dec(P)

% --- Rastrigin Decimal Fitness Function ---
%
%   [out] = rastrigin_fitness(P)
%
%   Input:
%       P = population of subjects                          [Nc x Ni]
%   Output:
%       OUT.
%           X = phenotype of each individual                [Nc x Ni]
%           F = fitness of each individual                  [1 x N1]
%           Fn = Normalized fitness of each individual      [1 x N1]

%% INITIALIZATIONS

[~,Ni] = size(P);       % Number of individuals

F = zeros(1,Ni);        % Fitness value for each individual

%% ALGORITHM

% Calculate fitness
for i = 1:Ni,
    F(i) = P(1,i)^2 + P(2,i)^2 + 20 - 10*(cos(2*pi*P(1,i)) + cos(2*pi*P(2,i)));
end

% Calculate Normalized Fitness
sig2n = 0.001;
Fn = (1./F + sig2n)/sum(1./F + sig2n);

%% FILL OUTPUT STRUCTURE

OUT.X = P;
OUT.F = F;
OUT.Fn = Fn;

%% END