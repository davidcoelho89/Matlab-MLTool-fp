function restriction = restriction_par_final(DATA,PAR,class_train)

% --- Verify a restriction for a combination of Parameters ---
%
%  restriction = restriction_par_final(DATA,PAR,class_train)
%
%   Input:
%       DATA = dataset                                          [struct]
%       PAR = struct including algorithm's hyperparameters      [struct]
%       class_train = handle of classifier training function 	[handle]
%   Output:
%       restriction: indicates the presence of a restriction    [0 or 1] 

%% INTIALIZATIONS

% Get Algorithm Name
algorithm_name = func2str(class_train);

% Init Output
restriction = 0;

%% ALGORITHM

% Restrictions for spok
if (strcmp(algorithm_name,'spok_train'))
    % Get Data
    [Nc,~] = size(DATA.output);
    % Get parameters
    [~,Nk] = size(PAR.Cx);
    max_prot = PAR.max_prot;
    % Verify Minimum number of prototypes
    if (Nk <= Nc)
        restriction = 1;
    end
    % Verify Maximum number of prototypes
    if (Nk >= max_prot)
        restriction = 1;
    end
end

%% FILL OUTPUT STRUCTURE

% Dont need.

%% END