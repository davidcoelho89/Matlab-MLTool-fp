function [PARout] = ps_train(DATA,PAR)

% --- PS classifier training ---
%
%   [PARout] = ps_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix                   [p x N]
%           output = labels matrix                      [Nc x N]
%       PAR.
%           Ne = maximum number of epochs               [cte]
%           eta = learning rate                         [0.01 0.1]
%           Von = enable or disable video               [cte]
%   Output:
%       PARout.
%           W = Regression / Classification Matrix      [Nc x p+1]
%           MQEtr = mean quantization error of training	[1 x Ne]
%           VID = frame structure (can be played with 'video function')

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Ne = 200;       	% maximum number of training epochs
    PARaux.eta = 0.05;    	% Learning step
    PARaux.Von = 0;         % disable video 
    PAR = PARaux;
else
    if (~(isfield(PAR,'Ne'))),
        PAR.Ne = 200;
    end
    if (~(isfield(PAR,'eta'))),
        PAR.eta = 0.05;
    end
    if (~(isfield(PAR,'Von'))),
        PAR.Von = 0;
    end
end

%% INITIALIZATIONS

% Data matrix
X = DATA.input;             % input matrix
D = DATA.output;            % output matrix (desired values)

% Hyperparameters Init
Nep = PAR.Ne;              	% maximum number of epochs
eta = PAR.eta;              % learning rate
Von = PAR.Von;              % enable or disable video

% Problem Init
[Nc,~] = size(D);           % Number of Classes (output neurons)
[p,N] = size(X);            % Number of attributes and samples
MQEtr = zeros(1,Nep);      	% Learning Curve

% Weight Matrix Init
if (isfield(PAR,'W')), 
    W = PAR.W;              % if already initialized
else
    W = 0.01*rand(Nc,p+1);  % else, initialize with small random numbers
end

% Add bias to input matrix [x0 = +1]
X = [ones(1,N);X];          

% Initialize Video Structure
VID = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

for ep = 1:Nep,

    % Save frame of the current epoch
    if (Von),
        VID(ep) = get_frame_hyperplane_lin(DATA,W);
    end
    
    % Shuffle Data
    I = randperm(N); 
    X = X(:,I); 
    D = D(:,I);   
    
    % Init sum of quadratic errors
    SQE = 0;
    
    for t = 1:N,   % 1 epoch
        
        % Calculate Output
        x = X(:,t);         % Get sample vector
        U = W*x;         	% Neurons Output vector
        Y = sign(U);      	% Neurons Activation Function (signal)
        
        % Error Calculation
        Er = D(:,t) - Y;    	% Quantization error or each neuron
        SQE = SQE + sum(Er.^2);	% sum of quadratic errors
        
        % Atualização dos Pesos
        W = W + eta*Er*x';
        
    end % end of one epoch
    
    % Mean squared quantization error for each epoch
    MQEtr(ep) = SQE/N;

end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;
PARout.MQEtr = MQEtr;
PARout.VID = VID;

%% END