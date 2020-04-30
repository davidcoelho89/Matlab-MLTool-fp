function [PAR] = isk2nn_dict_updt(DATA,HP)

% --- Procedure for Dictionary Prunning ---
%
%   [PAR] = isk2nn_dict_updt(DATA,HP)
%
%   Input:
%       DATA.
%           xt = attributes of sample                           [p x 1]
%           yt = class of sample                                [Nc x 1]
%       PAR.
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
%           Us = Update strategy                                [cte]
%               = 0 -> do not update prototypes
%               = 1 -> lms (wta)
%               = 2 -> lvq (supervised)
%               = 3 -> ng (neural gas - neigborhood)
%               = 4 -> som (self-organizing maps - neigborhood)
%           eta = Update rate                                   [cte]
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

Dm = HP.Dm;                         % Design Method
Us = HP.Us;                         % Update Strategy
sig2n = HP.sig2n;                   % Kernel Regularization parameter
eta = HP.eta;                       % Update rate

% Get Parameters

Dx = HP.Cx;                         % Attributes of dictionary
Dy = HP.Cy;                         % Classes of dictionary
Km = HP.Km;                         % Dictionary Kernel Matrix
Kmc = HP.Kmc;                       % Dictionary Kernel Matrix (class)
Kinv = HP.Kinv;                     % Dictionary Inverse Kernel Matrix
Kinvc = HP.Kinvc;                   % Dictionary Inv Kernel Matrix (class)
score = HP.score;                   % Score of each prototype
class_history = HP.class_history; 	% Classification history of each prototype
times_selected = HP.times_selected; % Prototypes # of selection

% Get Data

xt = DATA.input;
yt = DATA.output;

% Get problem parameters

[~,m] = size(Dx);   % hold dictionary size

%% 1 DICTIONARY FOR ALL DATA SET

if (Dm == 1 && Us ~= 0),
    
    % Find nearest prototype

    win = prototypes_win(Dx,xt,HP);
    
    % Update Closest Prototype (winner)
    
    if (Us == 1),       % (WTA)
        Dx(:,win) = Dx(:,win) + eta * (xt - Dx(:,win));
    elseif (Us == 2),	% (LVQ)
        [~,yt_i] = max(yt);
        [~,Dy_win] = max(Dy(:,win));
        if (yt_i == Dy_win),
            Dx(:,win) = Dx(:,win) + eta * (xt - Dx(:,win));
        else
            Dx(:,win) = Dx(:,win) - eta * (xt - Dx(:,win));
        end
    elseif (Us == 3), % (NG)
        % ToDo - All
    elseif (Us == 4), % (SOM)
        % ToDo - All
    end
    
    % Permute Postions of winner and last prototype
    
    Dx(:,[win,m]) = Dx(:,[m,win]);
    Dy(:,[win,m]) = Dy(:,[m,win]);

    % Remove line and column from kernel matrix
    
    Kpq = Km;
    Kpq(win,:) = [];
    Kpq(:,win) = [];
    
    % Remove line and column from inverse kernel matrix
    
    ep = zeros(m,1);
    ep(win) = 1;
    
    u = Km(:,win) - ep;
    
    eq = zeros(m,1);
    eq(win) = 1;
    
    v = eq;
    
    Kpq_inv = Kinv + (Kinv * u)*(v' * Kinv) / (1 - v' * Kinv * u);

    Kpq_inv(win,:) = [];
    Kpq_inv(:,win) = [];
    
    % Add line and column to kernel matrix
    
    % Calculate kt
    kt = zeros(m-1,1);
    for i = 1:m-1,
        kt(i) = kernel_func(Dx(:,i),Dx(:,m),HP);
    end
    
    % Calculate ktt
    ktt = kernel_func(Dx(:,m),Dx(:,m),HP);
    
    % Expand Kernel Matrix (and its inverse)
    Km = [Kpq, kt; kt', ktt + sig2n];
    
    % Add line and column to inverse kernel matrix
    
    % Calculate coefficients
    at = Kpq_inv*kt;
        
    % Calculate gamma
    gamma = sig2n + ktt - kt'*at;
    
    % Expand Inverse Kernel Matrix
    Kinv = (1/gamma)*[gamma*Kpq_inv + at*at', -at; -at', 1];
    
end

%% 1 DICTIONARY FOR EACH CLASS

if (Dm == 2 && Us ~= 0),
    
    % Get sample class and dictionary labels in sequential pattern
    
    [~,c] = max(yt);           	% get sequential class of sample
    [~,Dy_seq] = max(Dy);   	% get sequential classes of dictionary
	mc = sum(Dy_seq == c);      % number of prototypes from class c
    
    % Get inputs and outputs of class c
  	
    Dx_c = Dx(:,Dy_seq == c);
 	Dy_c = Dy(:,Dy_seq == c);

    % Find nearest prototype

    win_c = prototypes_win(Dx_c,xt,HP);
    win = prototypes_win(Dx,Dx_c(:,win_c),HP);    
    
    % Update Closest Prototype (winner)
    
    if (Us == 1),       % (WTA)
        Dx_c(:,win_c) = Dx_c(:,win_c) + eta * (xt - Dx_c(:,win_c));
    elseif (Us == 2),	% (LVQ)
        [~,yt_i] = max(yt);
        [~,Dy_win] = max(Dy_c(:,win_c));
        if (yt_i == Dy_win),
            Dx_c(:,win_c) = Dx_c(:,win_c) + eta * (xt - Dx_c(:,win_c));
        else
            Dx_c(:,win_c) = Dx_c(:,win_c) - eta * (xt - Dx_c(:,win_c));
        end
    elseif (Us == 3),   % NG
        % ToDo - All
    elseif (Us == 4),   % SOM
        % ToDo - All
    end
    
    Dx(:,win) = Dx_c(:,win_c);
    
    % Permute Postions of winner and last prototype
    
    Dx_c(:,[win_c,mc]) = Dx_c(:,[mc,win_c]);
    %Dy_c(:,[win_c,mc]) = Dy_c(:,[mc,win_c]);
        
    Dx(:,[win,m]) = Dx(:,[m,win]);
    Dy(:,[win,m]) = Dy(:,[m,win]);
    score(:,[win,m]) = score(:,[m,win]);
    class_history(:,[win,m]) = class_history(:,[m,win]);
    times_selected(:,[win,m]) = times_selected(:,[m,win]);
    
    % Remove line and column from kernel matrix
    
    Kpq_c = Km{c};
    Kpq_c(win_c,:) = [];
    Kpq_c(:,win_c) = [];
    
    % Remove line and column from inverse kernel matrix
    
    ep = zeros(mc,1);
    ep(win_c) = 1;
    
    u = Km{c}(:,win_c) - ep;
    
    eq = zeros(mc,1);
    eq(win_c) = 1;
    
    v = eq;
    
    Kpq_inv_c = Kinv{c} + (Kinv{c}*u)*(v'*Kinv{c}) / (1 - v'*Kinv{c}*u);

    Kpq_inv_c(win_c,:) = [];
    Kpq_inv_c(:,win_c) = [];
    
    % Add line and column to kernel matrix
    
    % Calculate kt
    kt = zeros(mc-1,1);
    for i = 1:mc-1,
        kt(i) = kernel_func(Dx_c(:,i),Dx_c(:,mc),HP);
    end
    
    % Calculate ktt
    ktt = kernel_func(Dx_c(:,mc),Dx_c(:,mc),HP);
    
    % Expand Kernel Matrix (and its inverse)
    Km{c} = [Kpq_c, kt; kt', ktt + sig2n];
    
	% Add line and column to inverse kernel matrix
    
    % Calculate coefficients
    at = Kpq_inv_c*kt;
        
    % Calculate gamma
    gamma = sig2n + ktt - kt'*at;
    
    % Expand Inverse Kernel Matrix
    Kinv{c} = (1/gamma)*[gamma*Kpq_inv_c + at*at', -at; -at', 1];
    
end

%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.Cx = Dx;
PAR.Cy = Dy;
PAR.Km = Km;
PAR.Kmc = Kmc;
PAR.Kinv = Kinv;
PAR.Kinvc = Kinvc;
PAR.score = score;
PAR.class_history = class_history;
PAR.times_selected = times_selected;

%% END