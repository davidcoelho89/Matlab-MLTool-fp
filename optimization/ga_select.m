function S = ga_select(P,Fn,Ss)

% --- Select Individuals for crossing ---
%
% 	S = ga_select(P,Fn,Ss)
%
%   Input:
%       P = population of subjects                  	[Nc*Cl x Ni]
%       Fn = Normalized fitness of each individual      [1 x N1]
%       Ss = Selection Strategy                        	[cte]
%
%   Output:
%       S = selected individuals for crossing           [1 x Ni]

%% INITIALIZATIONS

% Get number of individuals and genes
[~,Ni] = size(P);

% Init selected 
S = zeros(1,Ni);

%% ALGORITHM

% Tournament Strategy
if Ss == 1,
    
    for i = 1:Ni/2,
        
        % aleatory order of Ni numbers
        I = randperm(Ni);
        
        % Get first subject selected for crossing
        if Fn(I(1)) > Fn(I(2)),
            S((2*i)-1) = I(1);
        else
            S((2*i)-1) = I(2);
        end
        
        % Get second subject selected for crossing
        if Fn(I(3)) > Fn(I(4)),
            S(2*i) = I(3);
        else
            S(2*i) = I(4);
        end
        
    end
    
end

%% FILL OUTPUT STRUCTURE



%% END