function [PAR] = isk2nn_dict_grow(DATA,HP)

% --- Sparsification Procedure for Increasing the Dictionary ---
%
%   [PAR] = isk2nn_dict_grow(DATA,HP)
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
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sig2n = kernel regularization parameter             [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%       OUT.
%           y_h = classifier's output                           [Nc x 1]
%           win = closest prototype to sample                   [1 x 1]
%           dist = distance of sample from each prototype       [Nk x 1]
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

% Get Dictionary Parameters

Dx = HP.Cx;           	% Attributes of dictionary
Dy = HP.Cy;           	% Classes of dictionary
Kinv = HP.Kinv;       	% Dictionary Inv Kernel Matrix (total)
Kinvc = HP.Kinvc;      	% Dictionary Inv Kernel Matrix (class)

% Get Data

xt = DATA.input;
yt = DATA.output;

% Get problem parameters

[~,m] = size(Dx);       % Dictionary size

[~,c] = max(yt);        % Class of sample

[~,Dy_seq] = max(Dy);	% Sequential classes of dictionary

mc = sum(Dy_seq == c);	% Number of prototypes from samples' class

%% 1 DICTIONARY FOR ALL DATA SET

if Dm == 1, 
    
    % Add first element to dictionary
    if (m == 0),
        D = isk2nn_add_sample(DATA,HP,1);
    else
        % Get criterion result
        if Ss == 1,
            OUTcrit = ald_criterion(Dx,xt,Kinv,HP);
        elseif Ss == 2,
            OUTcrit = coherence_criterion(Dx,xt,HP);
        elseif Ss == 3,
            OUTcrit = novelty_criterion(Dx,Dy,xt,yt,HP);
        elseif Ss == 4,
            OUTcrit = surprise_criterion(Dx,Dy,xt,yt,Kinv,HP);
        end
        criterion_result = OUTcrit.result;
        
        % Expand or not Dictionary
        D = isk2nn_add_sample(DATA,HP,criterion_result);
    end
end

%% 1 DICTIONARY FOR EACH CLASS

if Dm == 2,
    
    % First Element of all dictionaries or of a class dictionary
    if (m == 0 || mc == 0),
        D = isk2nn_add_sample(DATA,HP,1);
    else
        % Get inputs and outputs from class c
        Dx_c = Dx(:,Dy_seq == c);
        Dy_c = Dy(:,Dy_seq == c);
        
        % Get criterion result
        if Ss == 1,
            OUTcrit = ald_criterion(Dx_c,xt,Kinvc{c},HP);
        elseif Ss == 2,
            OUTcrit = coherence_criterion(Dx_c,xt,HP);
        elseif Ss == 3,
            OUTcrit = novelty_criterion(Dx_c,Dy_c,xt,yt,HP);
        % Surprise
        elseif Ss == 4,
            OUTcrit = surprise_criterion(Dx_c,Dy_c,xt,yt,Kinvc{c},HP);
        end % end of Ss
        criterion_result = OUTcrit.result;
        
        % Expand or not dictionary
        D = isk2nn_add_sample(DATA,HP,criterion_result);
    end
    
end % end of Dm == 2
    
%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.Cx = D.Cx;
PAR.Cy = D.Cy;
PAR.Km = D.Km;
PAR.Kmc = D.Kmc;
PAR.Kinv = D.Kinv;
PAR.Kinvc = D.Kinvc;
PAR.score = D.score;
PAR.class_history = D.class_history;
PAR.times_selected = D.times_selected;

%% END