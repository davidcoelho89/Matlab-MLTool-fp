function [COEHout] = coherence_criterion(Dx,xt,HP)

% --- Apply the Coherence Criterion between a dictionary and a sample ---
%
%   [COEHout] = coherence_criterion(Dx,xt,Kinv,HP)
%
%   Input:
%       Dx = dictionary prototypes' inputs                      [p x Nk]
%       xt = sample to be tested                                [p x 1]
%       
%   Output:
%       COEHout.
%           result = if a sample fulfill the test               [0 or 1]
%           u_max = coherence measure                           [cte]

%% INITIALIZATIONS

[~,m] = size(Dx);               % Dictionary size
v1 = HP.v1;                 	% Sparsification parameter

%% ALGORITHM

% Init coherence measure (first element of dictionary)
u = kernel_func(Dx(:,1),xt,HP) / ...
    (sqrt(kernel_func(Dx(:,1),Dx(:,1),HP) * ...
    kernel_func(xt,xt,HP)));
u_max = abs(u);

% Get coherence measure
if (m >= 2),
    for i = 2:m,
        % Calculate kernel
        u = kernel_func(Dx(:,i),xt,HP) / ...
            (sqrt(kernel_func(Dx(:,i),Dx(:,i),HP) * ...
            kernel_func(xt,xt,HP)));
        % Calculate Coherence
        if (abs(u) > u_max),
            u_max = abs(u);
        end
    end
end

% Calculate Criterion
result = (u_max <= v1);

%% FILL OUTPUT STRUCTURE

COEHout.result = result;
COEHout.u_max = u_max;

%% END