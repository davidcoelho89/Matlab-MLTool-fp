function [OUT] = lssvc_classify(DATA,PAR)

% --- LSSVC classifier test ---
%
%   [OUT] = lssvc_classify(DATA,PAR)
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

%% ALGORITHM

[OUT] = svc_classify(DATA,PAR);

%% END