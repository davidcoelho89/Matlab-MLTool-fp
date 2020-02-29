function [PARout] = rls_train(DATA,PAR)

% --- RLS Classifier Training ---
%
%   [PARout] = rls_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes                          [p x N]
%           output = labels                             [Nc x N]
%       PAR.
%           lambda = forgiving factor [0.9 to 1]        [cte] 
%           Von = enable or disable video [0 or 1]    	[cte] 
%   Output:
%       PARout.
%           W = Regression / Classification Matrix      [Nc x p+1]
%           VID = frame structure (can be played with 'video function')

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.lambda = 1;      % dont have forgiving factor
    PARaux.Von = 0;         % disable video 
    PAR = PARaux;
else
    if (~(isfield(PAR,'lambda'))),
        PAR.lambda = 1;
    end
    if (~(isfield(PAR,'Von'))),
        PAR.Von = 0;
    end
end    

%% INITIALIZATIONS

% Data Initialization
X = DATA.input;             % Input Matrix
Y = DATA.output;            % Output Matrix

% Hyperparameters Initialization
lambda = PAR.lambda;        % forgiving factor
Von = PAR.Von;              % enable or disable video

% Problem Initialization
[p,~] = size(X);            % Number of attributes
[No,N] = size(Y);           % Number of outputs and samples

% Weight Matrix Init
if (isfield(PAR,'W')), 
    W = PAR.W;              % if already initialized
else
    W = zeros(p+1,No);   	% W0; could be W = 0.01*rand(p,No);
end

% P Matrix Init
if (isfield(PAR,'P')), 
    P = PAR.P;              % if already initialized
else
    P = 1e+4 * eye(p+1);    % P0;
end

% Add bias to input matrix
X = [ones(1,N);X];  % x0 = +1

% Initialize Video Structure
VID = struct('cdata',cell(1,N),'colormap', cell(1,N));

%% ALGORITHM

for i = 1:N,
    
    % Get sample (input and output)
    x = X(:,i);
    y = Y(:,i);

    % Update weights
    K = P*x/(lambda + x'*P*x);
    error = (y' - x'*W);
    W = W + K*error;
    P = (1/lambda)*(P - K*x'*P);
    
    % Save frame of the current epoch
    if (Von),
        VID(i) = hyperplane_frame(W',DATA);
    end
    
end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W';
PARout.VID = VID;

%% END