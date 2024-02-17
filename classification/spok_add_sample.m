function [PAR] = spok_add_sample(DATAn,HP)

% --- Add a Sample to Dictionary and Update its Variables ---
%
%   [PAR] = spok_add_sample(DATAn,HP)
%
%   Input:
%       DATAn.
%           input = attributes of sample                     	[p x 1]
%           output = class of sample                        	[Nc x 1]
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
%           sig2n = kernel regularization parameter             [cte]
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

% Get Data

xt = DATAn.input;
yt = DATAn.output;

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

ktt = kernel_func(xt,xt,HP);    % Kernel function of sample and itself
[Nc,~] = size(yt);              % Number of classes
[~,c] = max(yt);                % Class of sample
[~,Q] = size(Dx);               % Number of prototypes in the Dictionary
[~,Dy_seq] = max(Dy);           % Sequential classes of dictionary

%% ALGORITHM

% Update Kernel Matrices (just if needed)

if(HP.update_kernel_matrix)

    if (Q == 0) % First element of dictionary

        % Build Kernel matrix and its inverse for each class
        Kmc_out = cell(Nc,1);
        Kmc_out{c} = ktt + sig2n;
        Kinvc_out = cell(Nc,1);
        Kinvc_out{c} = 1/Kmc_out{c};

        % Build Kernel matrix and its inverse for dataset
        Km_out = ktt + sig2n;
        Kinv_out = 1/Km_out;

    else

        % Get number of prototypes from samples' class
        Qc = sum(Dy_seq == c);	

        % Build kernel matrix and its inverse of samples' class
        if (Qc == 0)
            Kmc{c} = ktt + sig2n;
            Kmc_out = Kmc;
            Kinvc{c} = 1/Kmc{c};
            Kinvc_out = Kinvc;

        % Update kernel matrix and its inverse of samples' class
        else
            % Get auxiliary variables
            Dx_c = Dx(:,Dy_seq == c);       % Inputs from class c
            kt_c = kernel_vect(Dx_c,xt,HP);
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
        kt = kernel_vect(Dx,xt,HP);
        at = Kinv*kt;
        delta = (ktt - kt'*at) + sig2n;

        % Update kernel matrix and its inverse for dataset
        Km_out = [Km, kt; kt', ktt + sig2n];
        Kinv_out = (1/delta)*[delta*Kinv + at*at', -at; -at', 1];
    end
    
end

% Add sample to dictionary

Cx_out = [Dx, xt];
Cy_out = [Dy, yt];

% Add variables used to prunning

score_out = [score,0];
class_history_out = [class_history,0];
times_selected_out = [times_selected,0];

%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.Cx = Cx_out;
PAR.Cy = Cy_out;
if(HP.update_kernel_matrix)
    PAR.Km = Km_out;
    PAR.Kmc = Kmc_out;
    PAR.Kinv = Kinv_out;
    PAR.Kinvc = Kinvc_out;
end
PAR.score = score_out;
PAR.class_history = class_history_out;
PAR.times_selected = times_selected_out;

%% END