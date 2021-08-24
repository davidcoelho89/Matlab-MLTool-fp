function restriction = gs_restriction_par_training(PAR,f_train)

% --- Verify a restriction for a combination of Parameters ---
%
%   restriction = gs_restriction_par_training(PAR,f_train)
%
%   Input:
%       PAR = struct including algorithm's hyperparameters      [struct]
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
if (strcmp(algorithm_name,'isk2nn_train'))
    % Get parameters
    [~,Nk] = size(PAR.Cx);
    max_prot = PAR.max_prot;
    % Verify Maximum number of prototypes
    if (Nk >= max_prot)
        restriction = 1;
    end
end

%% FILL OUTPUT STRUCTURE

% Dont need.

%% END