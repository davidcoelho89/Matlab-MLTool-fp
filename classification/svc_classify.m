function [OUT] = svc_classify(DATA,PAR)

% --- SVC classifier training ---
%
%   [OUT] = svc_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = test data attributes                        [p x N]
%       PAR.
%           Xsv = attributes of support vectors                 [p x Nsv]
%           Ysv = labels of support vectors                     [Nc x Nsv]
%           alpha = langrage multiplier                         [Nc x Nsv]
%           b0 = optimum bias                                   [Nc x 1]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       OUT.
%           y_h = classifier's output                           [Nc x N]

%% INITIALIZATIONS

% Get Testing Data
Xts = DATA.input;

% Initialize Problem
[~,Nts] = size(Xts);    % number of test samples
[~,Nc] = size(PAR.Ysv); % number of classes

% Initialize Output
y_h = zeros(Nc,Nts);

%% ALGORITHM

% Calculate estimated output for each class

for c = 1:Nc,
    
    % if it is a binary classifier, calculates the outputs only once
    
    if (c == 2 && Nc == 2),
        
        y_h(2,:) = -y_h(1,:);
        
    else
        
        alpha = PAR.alpha{c};   % get lagrange multipliers
        Xsv = PAR.Xsv{c};       % get attributes of support vectors
        Ysv = PAR.Ysv{c};       % get labels of support vectors
        b0 = PAR.b0{c};         % get optimum bias
        [~,Nsv] = size(Xsv);    % number of support vectors
        
        for i = 1:Nts,
            
            % Calculate W' * Xts
            K = zeros(1,Nsv);
            for j = 1:Nsv,
                K(j) = alpha(j)*Ysv(j)*kernel_func(Xsv(:,j),Xts(:,i),PAR);
            end
            Wx = sum(K);
            
            % Calculate Output
            y_h(c,i) = Wx + b0;
        end
        
    end
    
end

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;

%% END