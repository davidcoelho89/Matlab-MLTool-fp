function [PAR] = spok_dict_prun(HP)

% --- Procedure for Dictionary Pruning ---
%
%   [PAR] = spok_dict_prun(HP)
%
%   Input:
%       HP.
%           Cx = Attributes of input dictionary                 [p x Nk]
%           Cy = Classes of input dictionary                    [Nc x Nk]
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Nk]
%           class_history = used for prunning method           	[1 x Nk]
%           times_selected = used for prunning method           [1 x Nk]
%           Ps = Prunning strategy                              [cte]
%               = 0 -> do not remove prototypes
%               = 1 -> score-based method 1
%               = 2 -> score-based method 2
%           min_score = score that leads to prune prototype     [cte]
%   Output: 
%       PAR.
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Nk]
%           class_history = used for prunning method           	[1 x Nk]
%           times_selected = used for prunning method           [1 x Nk]

%% INITIALIZATIONS

% Get Hyperparameters

Ps = HP.Ps;                	% Pruning Strategy
min_score = HP.min_score; 	% Score that leads the prototype to be pruned

% Get Parameters

Dy = HP.Cy;              	% Classes of dictionary
score = HP.score;          	% Score of each prototype

% Get problem parameters

[~,m] = size(Dy);           % hold dictionary size

%% ALGORITHM
   
if (Ps == 0)

    % Does nothing

elseif (Ps == 1 || Ps == 2)

    [~,Dy_seq] = max(Dy);	% get sequential label of dictionary

    for k = 1:m
        if (score(k) < min_score)
            
            % number of elements from the same class as the prototypes'
            c = Dy_seq(k);
            mc = sum(Dy_seq == c);

            % dont rem element if it is the only element of its class
            if (mc == 1)
                continue;
            end
            
            % Remove Prototype from dictionary (just one per loop)
            HP = spok_rem_sample(HP,k);
            break;
        end
    end

end

%% FILL OUTPUT STRUCTURE

PAR = HP;

%% END