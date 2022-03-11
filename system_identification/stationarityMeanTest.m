function isStationary = stationarityMeanTest(x,nparts,tol)

% --- Verify Mean Condition for Stationarity ---
%
%   isStationary = stationarityMeanTest(x,nparts,tol)
%
%   Input:
%       x = Sampled signal                          [1 x Ns]
%       nparts = number of new vectors              [cte]
%       tol = used to indicate mean stationarity    [cte]
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

% Calculate mean of each part and get minimum and maximum
mean_x = mean(X,2);
min_mean = min(mean_x);
max_mean = max(mean_x);

disp(abs(max_mean - min_mean));

% Verify if difference of means exceed the tolerance
if (abs(max_mean - min_mean) < tol)
    isStationary = 1;
else
    isStationary = 0;
end

%% END