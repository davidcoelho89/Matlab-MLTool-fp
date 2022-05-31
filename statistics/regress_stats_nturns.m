function [nSTATS] = regress_stats_nturns(STATS_acc)

% --- Provide Statistics of n turns of Regression ---
%
%   function [nSTATS] = regress_stats_nturns(STATS_acc)
% 
%   Input:
%    	STATS_acc = Cell containing statistics of n turns of regression
%   Output:
%       nSTATS.
%           rmse = RMSE vector of all turns [1 x t]
%           rmse_max = Maximum RMSE obtained
%           rmse_max_index = index of Maximum RMSE
%           rmse_min = Minimum RMSE obtained
%           rmse_min_index = index of Minimum RMSE
%           rmse_mean = mean RMSE
%           rmse_median = median RMSE
%           rmse_std = standard deviation of RMSE
%           rmse_cv = Coefficient of Variation of RMSE


%% INITIALIZATIONS

% Get number of turns
[t,~] = size(STATS_acc);

rmse = zeros(1,t);           % RMSE vector of all turns [1 x t]
rmse_max = 0;                % Maximum RMSE obtained
rmse_max_index = 1;          % index of Maximum RMSE
rmse_min = 1;                % Minimum RMSE obtained
rmse_min_index = 1;          % index of Minimum RMSE
rmse_mean = 0;               % mean RMSE
rmse_median = 0;             % median RMSE
rmse_std = 0;                % standard deviation of RMSE
rmse_cv = 0;                 % Coefficient of Variation of RMSE

%% ALGORITHM

for i = 1:t
    STATS = STATS_acc{i};
    % Error vector
    rmse(i) = STATS.rmse;
    % Maximum error
    if (STATS.rmse > rmse_max)
        rmse_max_index = i;
        rmse_max = STATS.rmse;
    end
    % Minimum error
    if (STATS.rmse < rmse_min)
        rmse_min_index = i;
        rmse_min = STATS.rmse;
    end
end

% Error Statistics
rmse_mean = rmse_mean + mean(rmse);
rmse_median = rmse_median + median(rmse);
rmse_std = rmse_std + std(rmse);
rmse_cv = rmse_cv + (rmse_std / rmse_mean);

%% FILL OUTPUT STRUCTURE

nSTATS.rmse = rmse;
nSTATS.rmse_max = rmse_max;
nSTATS.rmse_max_index = rmse_max_index;
nSTATS.rmse_min = rmse_min;
nSTATS.rmse_min_index = rmse_min_index;
nSTATS.rmse_mean = rmse_mean;
nSTATS.rmse_median = rmse_median;
nSTATS.rmse_std = rmse_std;
nSTATS.rmse_cv = rmse_cv;

%% END