function isStationary = stationarityVarTest(x,nparts,tol)

% --- Verify Variance Condition for Stationarity ---
%
%   isStationary = stationarityVarTest(x,nparts,tol)
%
%   Input:
%       x = Sampled signal                          [1 x Ns]
%       nparts = number of new vectors              [cte]
%       tol = used to indicate var stationarity     [cte]
%   Output:
%       isStationary = Verification                 [0 or 1]

%% INITIALIZATIONS

Npoints = length(x);
parts_length = fix(Npoints/nparts);

%% ALGORITHM

% Divide signal in N parts
X = zeros(nparts,parts_length);
for i = 1:nparts
    X(i,:) = x( (i-1)*parts_length + 1 : (i)*parts_length );
end

% Calculate variance of each part and get minimum and maximum
var_x = var(X,0,2);
min_var = min(var_x);
max_var = max(var_x);

disp(abs(max_var - min_var));

% Verify if difference of variances exceed the tolerance
if (abs(max_var - min_var) < tol)
    isStationary = 1;
else
    isStationary = 0;
end

%% END