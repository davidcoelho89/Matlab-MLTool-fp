function [PAR] = spok_add_sample(DATA,HP)

% --- Add a Sample to Dictionary and Update its Variables ---
%
%   [PAR] = spok_add_sample(DATA,HP)
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
%           sig2n = kernel regularization parameter             [cte]
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

% Get Data

xt = DATA.input;
yt = DATA.output;

% Get Hyperparameters

sig2n = HP.sig2n;                   % Kernel regularization parameter

% Get Parameters

Dx = HP.Cx;                         % Attributes of dictionary
Dy = HP.Cy;                         % Classes of dictionary
Km = HP.Km;                         % Dictionary Kernel Matrix (total)
Kmc = HP.Kmc;                       % Dictionary Kernel Matrix (class)
Kinv = HP.Kinv;                     % Dictionary Inv Kernel Matrix (total)
Kinvc = HP.Kinvc;                   % Dictionary Inv Kernel Matrix (class)
score = HP.score;                   % Prototypes score for prunning
class_history = HP.class_history;	% Prototypes last classification
times_selected = HP.times_selected; % Prototypes # of selection

% Get problem parameters

[Nc,~] = size(yt);              % Number of classes
[~,c] = max(yt);                % Class of sample
[~,m] = size(Dx);               % Dictionary size
[~,Dy_seq] = max(Dy);           % Sequential classes of dictionary
mc = sum(Dy_seq == c);          % Number of prototypes from samples' class
ktt = kernel_func(xt,xt,HP);    % Kernel function of sample

%% ALGORITHM

% Add sample to dictionary
Cx_out = [Dx, xt];
Cy_out = [Dy, yt];

% Add variables used to prunning
score_out = [score,0];
class_history_out = [class_history,0];
times_selected_out = [times_selected,0];

% Update Kernel Matrices

if (m == 0)

    % Build Kernel matrix and its inverse for each class
    Kmc_out = cell(Nc,1);
    Kmc_out{c} = ktt + sig2n;
    Kinvc_out = cell(Nc,1);
    Kinvc_out{c} = 1/Kmc_out{c};

    % Build Kernel matrix and its inverse for dataset
    Km_out = ktt + sig2n;
    Kinv_out = 1/Km_out;

else

    % Build kernel matrix and its inverse of samples' class
    if (mc == 0)
        Kmc{c} = ktt + sig2n;
        Kmc_out = Kmc;
        Kinvc{c} = 1/Kmc{c};
        Kinvc_out = Kinvc;

    % Update kernel matrix and its inverse of samples' class
    else
        % Get inputs from class c
        Dx_c = Dx(:,Dy_seq == c);
        % Get auxiliary variables
        kt_c = zeros(mc,1);
        for i = 1:mc
            kt_c(i) = kernel_func(Dx_c(:,i),xt,HP);
        end
        at_c = Kinvc{c}*kt_c;
        delta_c = (ktt - kt_c'*at_c) + sig2n;
        % Update Kernel matrix
        Kmc{c} = [Kmc{c}, kt_c; kt_c', ktt + sig2n];
        Kmc_out = Kmc;
        % Update Inverse Kernel matrix
        Kinvc{c} = (1/delta_c)* ...
                   [delta_c*Kinvc{c} + at_c*at_c',-at_c;-at_c',1];
        Kinvc_out = Kinvc;
    end

    % Get auxiliary variables
    kt = zeros(m,1);
    for i = 1:m
        kt(i) = kernel_func(Dx(:,i),xt,HP);
    end
    at = Kinv*kt;
    delta = (ktt - kt'*at) + sig2n;

    % Update kernel matrix and its inverse for dataset
    Km_out = [Km, kt; kt', ktt + sig2n];
    Kinv_out = (1/delta)*[delta*Kinv + at*at', -at; -at', 1];
end

%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.Cx = Cx_out;
PAR.Cy = Cy_out;
PAR.Km = Km_out;
PAR.Kmc = Kmc_out;
PAR.Kinv = Kinv_out;
PAR.Kinvc = Kinvc_out;
PAR.score = score_out;
PAR.class_history = class_history_out;
PAR.times_selected = times_selected_out;

%% END