function [PAR] = spok_dict_grow(DATA,HP)

% --- Sparsification Procedure for Increasing the Dictionary ---
%
%   [PAR] = spok_dict_grow(DATA,HP)
%
%   Input:
%       DATA.
%           xt = attributes of sample                           [p x 1]
%           yt = class of sample                                [Nc x 1]
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
%           Dm = Design Method                                  [cte]
%               = 1 -> all data set
%               = 2 -> per class
%           Ss = Sparsification strategy                        [cte]
%               = 1 -> ALD
%               = 2 -> Coherence
%               = 3 -> Novelty
%               = 4 -> Surprise
%           v1 = Sparseness parameter 1                         [cte]
%           v2 = Sparseness parameter 2                         [cte]
%           max_prot = max number of prototypes ("Budget")      [cte]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sig2n = kernel regularization parameter             [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output: 
%       PAR.
%           Cx = Attributes of output dictionary                [p x Nk]
%           Cy = Classes of  output dictionary                  [Nc x Nk]
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Nk]
%           class_history = used for prunning method           	[1 x Nk]
%           times_selected = used for prunning method           [1 x Nk]

%% INITIALIZATIONS

% Get Hyperparameters

Dm = HP.Dm;           	% Design method
Ss = HP.Ss;            	% Sparsification Strategy
max_prot = HP.max_prot; % Max number of prototypes

% Get Dictionary Prototypes

Cx = HP.Cx;           	% Attributes of prototypes
Cy = HP.Cy;           	% Classes of prototypes

% Get Data

xt = DATA.input;
yt = DATA.output;

% Get problem parameters

[~,m] = size(Cx);       % Dictionary size

[~,c] = max(yt);        % Class of sample (Sequential encoding)

[~,Cy_seq] = max(Cy);	% Classes of dictionary (Sequential encoding)

mc = sum(Cy_seq == c);	% Number of prototypes from samples' class

%% ALGORITHM

% Add first element to dictionary (total or from class)
if (m == 0 || (Dm == 2 && mc == 0))
    
    HP = spok_add_sample(DATA,HP);
    
else
    % Dont add if number of prototypes is too high
    if (m < max_prot)
    	
        % Get Dictionary Samples and Inverse Kernel Matrix
        if (Dm == 1)
            Dx = Cx;
            Dy = Cy;
            Kinv = HP.Kinv;
        elseif (Dm == 2)
            Dx = Cx(:,Cy_seq == c);
            Dy = Cy(:,Cy_seq == c);
            Kinv = HP.Kinvc{c};
        end

        % Get criterion result
        if Ss == 1
            OUTcrit = ald_criterion(Dx,xt,HP,Kinv);
        elseif Ss == 2
            OUTcrit = coherence_criterion(Dx,xt,HP);
        elseif Ss == 3
            OUTcrit = novelty_criterion(Dx,Dy,xt,yt,HP);
        elseif Ss == 4
            OUTcrit = surprise_criterion(Dx,Dy,xt,yt,HP,Kinv);
        end

        % Expand or not Dictionary
        if(OUTcrit.result == 1)
            HP = spok_add_sample(DATA,HP);
        end
        
    end
end


%% FILL OUTPUT STRUCTURE

PAR = HP;

%% END