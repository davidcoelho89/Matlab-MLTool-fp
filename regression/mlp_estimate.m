function [PAR] = mlp_estimate(DATA,HP)

% --- MLP Regression Training ---
%
%   [PARout] = mlp_estimate(DATA,PAR)
%
%   Input:
%       DATA.
%           input = regression variables	[lag_u + lag_y x Ns-lag_max]
%           output = regression output   	[Ny  x  Ns-lag_max]
%           lag_input = lag_u             	[1 x Nu]
%           lag_output = lag_y            	[1 x Ny]
%       PAR.
%           Ne = maximum number of epochs	                    [cte]
%           Nh = number of hidden neurons                       [NL-1 x 1]
%           eta = learning rate           	                    [0.01 0.1]
%           mom = moment factor            	                    [0.5 1.0]
%           Nlin = non-linearity           	                    [cte]
%               1 -> Sigmoid                                    [0 e 1]
%               2 -> Hyperbolic Tangent                         [-1 e +1]
%           add_bias = add or not bias for input                [0 or 1]
%           Von = enable or disable video                       [cte]
%           prediction_type = type of prediction                [cte]
%               =0 -> Free simulation
%               >0 -> "n-steps ahead"
%   Output:
%       PARout.
%           W = weight matrices                                 [NL x 1]
%               W{1:end-1} = Hidden layer weight Matrix 
%               W{end} = Output layer weight Matrix
%           MQEtr = Mean Quantization Error (learning curve)	[Ne x 1]
%           VID = frame structure 
%               (can be played with 'video function')
%           lag_output = lag for each output                    [1 x Ny]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(HP)))
    PARaux.Nh = 10;             % Number of hidden neurons
    PARaux.Ne = 200;            % Maximum training epochs
    PARaux.eta = 0.05;          % Learning Step
    PARaux.mom = 0.75;          % Moment Factor
    PARaux.Nlin = 2;            % Non-linearity
    PARaux.add_bias = 1;        % Add bias to input
    PARaux.Von = 0;             % Disable video
    PARaux.prediction_type = 1; % 1-step ahead prediction
    HP = PARaux;
    
else
    if (~(isfield(HP,'Nh')))
        HP.Nh = 10;
    end
    if (~(isfield(HP,'Ne')))
        HP.Ne = 200;
    end
    if (~(isfield(HP,'eta')))
        HP.eta = 0.05;
    end
    if (~(isfield(HP,'mom')))
        HP.mom = 0.75;
    end
    if (~(isfield(HP,'Nlin')))
        HP.Nlin = 2;
    end
    if (~(isfield(HP,'add_bias')))
        HP.add_bias = 1;
    end
    if (~(isfield(HP,'add_bias')))
        HP.add_bias = 1;
    end
    if (~(isfield(HP,'Von')))
        HP.Von = 0;
    end
    if (~(isfield(HP,'prediction_type')))
        HP.prediction_type = 1;
    end
end

%% INITIALIZATIONS

% Data Initialization
X = DATA.input;             % Input Matrix
Y = DATA.output;            % Output Matrix

% Hyperparameters Initialization
Nep = HP.Ne;            	% Number of training epochs
Nh = HP.Nh;                % Number of hidden neurons
eta = HP.eta;            	% learning rate 
mom = HP.mom;            	% Moment Factor
Nlin = HP.Nlin;          	% Non-linearity
Von = HP.Von;            	% Enable or disable video
add_bias = HP.add_bias;    % Add bias to input (or not)

% Problem Initialization
[No,~] = size(Y);           % Number of Outputs (Regression and Neurons)
[p,N] = size(X);            % attributes and samples
MQEtr = zeros(1,Nep);     	% Mean Quantization Error
structure = [p,Nh,No];      % MLP Structure
NL = length(structure) - 1; % Number of layers

% disp('MLP Structure:');
% disp(structure);

% Cells for holding info of forward and backward steps
x = cell(NL,1);             % input of each layer
yh = cell(NL,1);         	% output of each layer
delta = cell(NL,1);         % local gradient of each layer

% Weight Matrices Initialization (NL-1 Hidden layers)

W = cell(NL,1);             % Weight Matrices
W_old = cell(NL,1);         % Necessary for moment factor

if (isfield(HP,'W'))       % if already initialized
    for i = 1:NL
        W{i} = HP.W{i};
    end
else                        % Initialize randomly
    for i = 1:NL
        W{i} = 0.01*rand(structure(i+1),structure(i)+1);
        W_old{i} = W{i};    
    end
end

% Add bias to input matrix
if(add_bias == 1)
    X = [ones(1,N) ; X]; 	% x0 = +1
end

% Initialize Video Structure
VID = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

for ep = 1:Nep   % for each epoch

    % Update Parameters
    HP.W = W;
    
    % Save frame of the current epoch
    if (Von)
        VID(ep) = get_frame_hyperplane(DATA,HP,@mlp_classify);
    end
    
    % Shuffle Data
    I = randperm(N);
    X = X(:,I);
    Y = Y(:,I);
    
    SQE = 0; % Init sum of quadratic errors
    
    for t = 1:N   % for each sample
        
        % Forward Step (Calculate Layers' Outputs)
        for i = 1:NL
            if (i == 1) % Input layer
                x{i} = X(:,t);               % Get input data
            else
                x{i} = [+1; yh{i-1}];        % add bias to last output
            end
            Ui = W{i} * x{i};           	 % Activation of hidden neurons
            
            if (i == NL)                     % Output layer
                yh{i} = mlp_f_ativ(Ui,0); 	 % Linear function
            else
                yh{i} = mlp_f_ativ(Ui,Nlin); % Non-linear function
            end
        end
        
        % For the controller: yh{NL} -> MODELO -> Ym
        
        % Error Calculation
        E = Y(:,t) - yh{NL};
        SQE = SQE + sum(E.^2);
        
        % Backward Step (Calculate Layers' Local Gradients)
        for i = NL:-1:1
            
            if (i == NL) % output layer
                f_der = mlp_f_gradlocal(yh{i},0);
                delta{i} = E.*f_der;
                % delta{i} = E;
            else
                f_der = mlp_f_gradlocal(yh{i},Nlin);
                delta{i} = f_der.*(W{i+1}(:,2:end)'*delta{i+1});
            end
        end
        
        % Weight Adjustment
        for i = NL:-1:1
            W_aux = W{i};
            W{i} = W{i} + eta*delta{i}*x{i}' + mom*(W{i} - W_old{i});
            W_old{i} = W_aux;
        end

    end   % end of epoch
        
    % Mean Squared Error of epoch
    MQEtr(ep) = SQE/N;
    
end   % end of all epochs

%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.W = W;
PAR.MQEtr = MQEtr;
PAR.VID = VID;
PAR.lag_output = DATA.lag_output;

%% END