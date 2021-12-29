function [PAR] = spok_rem_sample(HP,win)

% --- Remove a Sample from Dictionary and Update its Variables ---
%
%   [D] = spok_rem_sample(DATA,HP,criterion_result)
%
%   Input:
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
%       win = indicates which prototype should be removed       [cte]
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

% Get Dictionary Variables

Cx = HP.Cx;                         % Attributes of dictionary
Cy = HP.Cy;                         % Classes of dictionary
Km = HP.Km;                         % Dictionary Kernel Matrix
Kmc = HP.Kmc;                       % Dictionary Kernel Matrix (class)
Kinv = HP.Kinv;                     % Dictionary Inverse Kernel Matrix
Kinvc = HP.Kinvc;                   % Dictionary Inv Kernel Matrix (class)
score = HP.score;                   % Score of each prototype
class_history = HP.class_history; 	% Classification history of each prototype
times_selected = HP.times_selected; % Prototypes # of selection

% Get Problem Variables

[~,m] = size(Cx);                   % hold dictionary size
xt = Cx(:,win);                     % Get winner prototype input

[~,c] = max(Cy(:,win));             % Get winner prototype class
[~,Dy_seq] = max(Cy);               % Get sequential classes of entire dict

Dx_c = Cx(:,Dy_seq == c);           % Get dicionary of prototype's class
win_c = prototypes_win(Dx_c,xt,HP); % Get position of prototype in class c

mc = sum(Dy_seq == c);              % Get number of prototypes in class c

%% ALGORITHM

% Remove positions from inverse kernel matrix (entire dict)

ep = zeros(m,1);
ep(win) = 1;
u = Km(:,win) - ep;

eq = zeros(m,1);
eq(win) = 1;
v = eq;

Kinv = Kinv + (Kinv * u)*(v' * Kinv) / (1 - v' * Kinv * u);
Kinv(win,:) = [];
Kinv(:,win) = [];

% Remove positions from kernel matrix (entire dict)

Km(win,:) = [];
Km(:,win) = [];

% Remove positions from inverse kernel matrices (class dict)

ep = zeros(mc,1);
ep(win_c) = 1;
u = Kmc{c}(:,win_c) - ep;

eq = zeros(mc,1);
eq(win_c) = 1;
v = eq;

Kinvc{c} = Kinvc{c} + (Kinvc{c}*u)*(v'*Kinvc{c}) / (1 - v'*Kinvc{c}*u);
Kinvc{c}(win_c,:) = [];
Kinvc{c}(:,win_c) = [];

% Remove positions from kernel matrix (class dict)

Kmc{c}(win_c,:) = [];
Kmc{c}(:,win_c) = [];

% Remove sample from dictionary
Cx(:,win) = [];
Cy(:,win) = [];

% Remove variables used to prunning
score(:,win) = [];
class_history(:,win) = [];
times_selected(:,win) = [];

%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.Cx = Cx;
PAR.Cy = Cy;
PAR.Km = Km;
PAR.Kmc = Kmc;
PAR.Kinv = Kinv;
PAR.Kinvc = Kinvc;
PAR.score = score;
PAR.class_history = class_history;
PAR.times_selected = times_selected;

%% END