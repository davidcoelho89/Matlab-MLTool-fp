function [PAR] = spok_dict_prun(HP)

% --- Procedure for Dictionary Pruning ---
%
%   [PAR] = spok_dict_prun(HP)
%
%   Input:
%       HP.
%           Cx = Attributes of input dictionary                 [p x Q]
%           Cy = Classes of input dictionary                    [Nc x Q]
%           Km = Kernel matrix of dictionary                    [Q x Q]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Q x Q]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Q]
%           class_history = used for prunning method           	[1 x Q]
%           times_selected = used for prunning method           [1 x Q]
%           Ps = Prunning strategy                              [cte]
%               = 0 -> do not remove prototypes
%               = 1 -> score-based method 1
%               = 2 -> score-based method 2
%           min_score = score that leads to prune prototype     [cte]
%   Output: 
%       PAR.
%           Km = Kernel matrix of dictionary                    [Q x Q]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Q x Q]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Q]
%           class_history = used for prunning method           	[1 x Q]
%           times_selected = used for prunning method           [1 x Q]

%% INITIALIZATIONS

% Get Hyperparameters

Ps = HP.Ps;                	% Pruning Strategy
min_score = HP.min_score; 	% Score that leads the prototype to be pruned

% Get Parameters

Dy = HP.Cy;              	% Classes of dictionary
score = HP.score;          	% Score of each prototype

% Get problem parameters

[~,Q] = size(Dy);           % hold dictionary size

%% ALGORITHM
   
if (Ps == 0)

    % Does nothing

elseif (Ps == 1 || Ps == 2)

    [~,Dy_seq] = max(Dy);	% get sequential label of dictionary

    for q = 1:Q
        if (score(q) < min_score)
            
            % number of elements from the same class as the prototypes'
            c = Dy_seq(q);
            Qc = sum(Dy_seq == c);

            % dont rem element if it is the only element of its class
            if (Qc == 1)
                continue;
            end
            
            % Hold number of times the removed prototype was selected
            HP.times_selected_sum = HP.times_selected_sum + ...
                                    HP.times_selected(q);

            % Remove Prototype from dictionary (just one per loop)
            HP = spok_rem_sample(HP,q);

            % Just remove one prototype per iteration
            break;

        end
    end

end

%% FILL OUTPUT STRUCTURE

PAR = HP;

%% END