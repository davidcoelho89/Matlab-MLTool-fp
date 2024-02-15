function [PAR] = spok_dict_grow(DATAn,HP)

% --- Sparsification Procedure for Increasing the Dictionary ---
%
%   [PAR] = spok_dict_grow(DATAn,HP)
%
%   Input:
%       DATAn.
%           input = attributes of sample                      	[p x 1]
%           output = class of sample                            [Nc x 1]
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
%           Cx = Attributes of output dictionary                [p x Q]
%           Cy = Classes of  output dictionary                  [Nc x Q]
%           Km = Kernel matrix of dictionary                    [Q x Q]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Q x Q]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Q]
%           class_history = used for prunning method           	[1 x Q]
%           times_selected = used for prunning method           [1 x Q]

%% INITIALIZATIONS

% Get Hyperparameters

Dm = HP.Dm;           	% Design method
Ss = HP.Ss;            	% Sparsification Strategy
max_prot = HP.max_prot; % Max number of prototypes

% Get Dictionary Prototypes

Cx = HP.Cx;           	% Attributes of prototypes
Cy = HP.Cy;           	% Classes of prototypes

% Get Data

xt = DATAn.input;
yt = DATAn.output;

% Get problem parameters

[~,Q] = size(Cx);       % Dictionary size

[~,c] = max(yt);        % Class of sample (Sequential encoding)

[~,Cy_seq] = max(Cy);	% Classes of dictionary (Sequential encoding)

Qc = sum(Cy_seq == c);	% Number of prototypes from samples' class

%% ALGORITHM

% Add first element to dictionary (total or from class)
if (Q == 0 || (Dm == 2 && Qc == 0))
    
    HP = spok_add_sample(DATAn,HP);
    
else
    % Dont add if number of prototypes is too high
    if (Q < max_prot)
    	
        % Get Dictionary Samples and Inverse Kernel Matrix
        if (Dm == 1)
            Dx = Cx;
            Dy = Cy;
            if(HP.update_kernel_matrix)
                Kinv = HP.Kinv;
            else
                Kinv = [];
            end
        elseif (Dm == 2)
            Dx = Cx(:,Cy_seq == c);
            Dy = Cy(:,Cy_seq == c);
            if(HP.update_kernel_matrix)
                Kinv = HP.Kinvc{c};
            else
                Kinv = [];
            end
        end

        % Get criterion result
        if Ss == 1
            OUTcrit = ald_criterion(Dx,xt,HP,Kinv);
        elseif Ss == 2
            OUTcrit = coherence_criterion(Dx,xt,HP);
        elseif Ss == 3
            OUTcrit = novelty_criterion(Dx,xt,yt,HP);
        elseif Ss == 4
            OUTcrit = surprise_criterion(Dx,Dy,xt,yt,HP,Kinv);
        end

        % Expand or not Dictionary
        if(OUTcrit.result == 1)
            HP = spok_add_sample(DATAn,HP);
        end
        
    end
end

%% FILL OUTPUT STRUCTURE

PAR = HP;

%% END