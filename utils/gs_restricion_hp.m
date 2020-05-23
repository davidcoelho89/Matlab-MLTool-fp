function restriction = gs_restricion_hp(HP,f_train)

% --- Verify a restriction for a combination of Hyperparameters ---
%
%   [restriction] = gs_restricion_hp(HP,f_train)
%
%   Input:
%       HP = struct including algorithm's hyperparameters       [struct]
%       f_train = handle of classifier training function        [handle]
%   Output:
%       restriction: indicates the presence of a restriction    [0 or 1] 

%% INTIALIZATIONS

% Get Algorithm Name
algorithm_name = func2str(f_train);

% Init Output
restriction = 0;

%% ALGORITHM

% Restrictions for isk2nn
if (strcmp(algorithm_name,'isk2nn_train')),
    % Get hyperparameters
    v1 = HP.v1;
    v2 = HP.v2;
    Ss = HP.Ss;
    % Verify v2 and v1 for novelty and suprise methods
    if((v2 <= v1) && (Ss == 3 || Ss == 4)),
        restriction = 1;
    end
end

%% FILL OUTPUT STRUCTURE

% Dont need.

%% END