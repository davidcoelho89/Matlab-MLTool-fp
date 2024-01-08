function [nSTATS] = class_stats_nturns(STATS_acc)

% --- Provide Statistics of n turn of Classification ---
%
%   [nSTATS] = class_stats_nturns(STATS_acc)
% 
%   Input:
%    	STATS_acc = Cell containing statistics of n turns of classification
%   Output:
%       nSTATS.
%			Mconf_sum = sum of confusion matrix     [Nc x Nc]
%       	Mconf_mean = mean confusion matrix      [Nc x Nc]
%
%           acc = vector with % of accuracy         [1 x t]
%           acc_max = maximum accuracy              [cte]
%           acc_max_i = index of maximum accuracy   [cte]
%           acc_min = minimum accuracy              [cte]
%           acc_min_i =  index of minimum accuracy  [cte]
%           acc_mean = mean of accuracy             [cte]
%           acc_median = median of accuracy         [cte]
%           acc_std = standard dev of accuracy      [cte]
%           acc_cv = Coefficient of variation       [cte]
%
%           err = vector with % of error            [cte]
%           err_max = maximum accuracy              [cte]
%           err_max_i = index of maximum accuracy   [cte]
%           err_min = minimum accuracy              [cte]
%           err_min_i =  index of minimum accuracy  [cte]
%           err_mean = mean of accuracy             [cte]
%           err_median = median of error            [cte]
%           err_std = standard dev of accuracy      [cte]
%           err_cv = Coefficient of variation       [cte]
%
%           roc_t = threshold                       [1 x t]
%           roc_tpr = true positive rate            [1 x t]
%           (a.k.a. recall, sensitivity)
%           roc_spec = specificity                  [1 x t]
%           (a.k.a. true negative rate) 
%           roc_fpr = false positive rate           [1 x t]
%           roc_prec = precision                    [1 x t]
%           roc_rec = recall                        [1 x t]
%
%           fsc = f1-score                          [1 x t]
%
%           mcc = Matthews Correlation Coef         [1 x t]
%
%           auc = area under the curve              [1 x t]

%% INITIALIZATIONS

% Get number of turns, classes, threshold length
[t,~] = size(STATS_acc);
[~,Nc] = size(STATS_acc{1,1}.Mconf);

% Init Outputs

Mconf_sum = zeros(Nc,Nc);   % Sum of confusion matrices [Nc x Nc]
Mconf_mean = zeros(Nc,Nc);  % Mean confusion matrices [Nc x Nc]

acc = zeros(1,t);           % Accuracy vector of all turns [1 x t]
acc_max = 0;                % Maximum accuracy obtained
acc_max_index = 1;          % index of Maximum accuracy obtained
acc_min = 1;                % Minimum accuracy obtained
acc_min_index = 1;          % index of Minimum accuracy obtained
acc_mean = 0;               % mean accuracy rate
acc_median = 0;             % median accuracy rate
acc_std = 0;                % standard deviation of accuracy rate
acc_cv = 0;                 % Coefficient of Variation of accuracy rate

err = zeros(1,t);           % Error vector of all turns [1 x t]
err_max = 0;                % Maximum error obtained
err_max_index = 1;          % index of Maximum error obtained
err_min = 1;                % Minimum error obtained
err_min_index = 1;          % index of Minimum error obtained
err_mean = 0;               % mean error rate
err_median = 0;             % median error rate
err_std = 0;                % standard deviation of error rate
err_cv = 0;                 % Coefficient of Variation of error rate

fsc = zeros(1,t);           % fsc vector of all turns [1 x t]
fsc_max = 0;                % Maximum fsc obtained
fsc_max_index = 1;          % index of Maximum fsc obtained
fsc_min = 1;                % Minimum fsc obtained
fsc_min_index = 1;          % index of Minimum fsc obtained
fsc_mean = 0;               % mean fsc
fsc_median = 0;             % median fsc
fsc_std = 0;                % standard deviation of fsc
fsc_cv = 0;                 % Coefficient of Variation of fsc

mcc = zeros(1,t);           % mcc vector of all turns [1 x t]
mcc_max = 0;                % Maximum mcc obtained
mcc_max_index = 1;          % index of Maximum mcc obtained
mcc_min = 1;                % Minimum mcc obtained
mcc_min_index = 1;          % index of Minimum mcc obtained
mcc_mean = 0;               % mean mcc
mcc_median = 0;             % median mcc
mcc_std = 0;                % standard deviation of mcc
mcc_cv = 0;                 % Coefficient of Variation of mcc

if(isfield(STATS_acc{1},'roc_t'))

    roc_t = cell(1,t);          % threshold of roc curve
    roc_tpr = cell(1,t);        % true positive rate
    roc_spec = cell(1,t);       % specificity
    roc_fpr = cell(1,t);        % false positive rate
    roc_prec = cell(1,t);       % precision
    roc_rec = cell(1,t);        % recall
    auc = cell(1,t);            % area under the curve

end

%% ALGORITHM

% Gerenal Statistics
for i = 1:t
    STATS = STATS_acc{i};
    % Sum of Confusion Matrices
    Mconf_sum = Mconf_sum + STATS.Mconf;
    % Accuracy / Error / F-score / Mcc Vectors
	acc(i) = STATS.acc;
    err(i) = STATS.err;
    fsc(i) = STATS.fsc_macro;
    mcc(i) = STATS.mcc_multiclass;
    % Maximum Accuracy / Error / F-score / Mcc
    if (STATS.acc > acc_max)
        acc_max_index = i;
        acc_max = STATS.acc;
    end
    if (STATS.err > err_max)
        err_max_index = i;
        err_max = STATS.err;
    end
    if (STATS.fsc > fsc_max)
        fsc_max_index = i;
        fsc_max = STATS.fsc;
    end
    if (STATS.mcc > mcc_max)
        mcc_max_index = i;
        mcc_max = STATS.mcc;
    end
    % Minimum Accuracy / Error / F-score / Mcc
    if (STATS.acc < acc_min)
        acc_min_index = i;
        acc_min = STATS.acc;
    end
    if (STATS.err < err_min)
        err_min_index = i;
        err_min = STATS.err;
    end
    if (STATS.fsc < fsc_min)
        fsc_min_index = i;
        fsc_min = STATS.fsc;
    end
    if (STATS.mcc < mcc_min)
        mcc_min_index = i;
        mcc_min = STATS.mcc;
    end
    % ROC parameters
    if(isfield(STATS,'roc_t'))
        roc_t{i} = STATS.roc_t;
        roc_tpr{i} = STATS.roc_tpr;
        roc_spec{i} = STATS.roc_spec;
        roc_fpr{i}  = STATS.roc_fpr;
        roc_prec{i} = STATS.roc_prec;
        roc_rec{i} = STATS.roc_rec;
        auc{i} = STATS.auc;
        fsc{i} = STATS.fsc;
        mcc{i} = STATS.mcc;
    end
    
end

% Mean Confusion matrix
Mconf_mean = Mconf_mean + Mconf_sum / t;

% Accuracy Statistics
acc_mean = acc_mean + mean(acc);
acc_median = acc_median + median(acc);
acc_std = acc_std + std(acc);
acc_cv = acc_cv + (acc_std / acc_mean);

% Error Statistics
err_mean = err_mean + mean(err);
err_median = err_median + median(err);
err_std = err_std + std(err);
err_cv = err_cv + (err_std / err_mean);

% F-score Statistics
fsc_mean = fsc_mean + mean(fsc);
fsc_median = fsc_median + median(fsc);
fsc_std = fsc_std + std(fsc);
fsc_cv = fsc_cv + (fsc_std / fsc_mean);

% Mcc Statistics
mcc_mean = mcc_mean + mean(mcc);
mcc_median = mcc_median + median(mcc);
mcc_std = mcc_std + std(mcc);
mcc_cv = mcc_cv + (mcc_std / mcc_mean);


%% FILL OUTPUT STRUCTURE

nSTATS.Mconf_sum = Mconf_sum;
nSTATS.Mconf_mean = Mconf_mean;

nSTATS.acc = acc;
nSTATS.acc_max = acc_max;
nSTATS.acc_max_i = acc_max_index;
nSTATS.acc_min = acc_min;
nSTATS.acc_min_i = acc_min_index;
nSTATS.acc_mean = acc_mean;
nSTATS.acc_median = acc_median;
nSTATS.acc_std = acc_std;
nSTATS.acc_cv = acc_cv;

nSTATS.err = err;
nSTATS.err_max = err_max;
nSTATS.err_max_i = err_max_index;
nSTATS.err_min = err_min;
nSTATS.err_min_i = err_min_index;
nSTATS.err_mean = err_mean;
nSTATS.err_median = err_median;
nSTATS.err_std = err_std;
nSTATS.err_cv = err_cv;

nSTATS.fsc = fsc;
nSTATS.fsc_max = fsc_max;
nSTATS.fsc_max_i = fsc_max_index;
nSTATS.fsc_min = fsc_min;
nSTATS.fsc_min_i = fsc_min_index;
nSTATS.fsc_mean = fsc_mean;
nSTATS.fsc_median = fsc_median;
nSTATS.fsc_std = fsc_std;
nSTATS.fsc_cv = fsc_cv;

nSTATS.mcc = mcc;
nSTATS.mcc_max = mcc_max;
nSTATS.mcc_max_i = mcc_max_index;
nSTATS.mcc_min = mcc_min;
nSTATS.mcc_min_i = mcc_min_index;
nSTATS.mcc_mean = mcc_mean;
nSTATS.mcc_median = mcc_median;
nSTATS.mcc_std = mcc_std;
nSTATS.mcc_cv = mcc_cv;

if(isfield(STATS_acc{1},'roc_t'))
    nSTATS.roc_t = roc_t;
    nSTATS.roc_tpr = roc_tpr;
    nSTATS.roc_spec = roc_spec;
    nSTATS.roc_fpr = roc_fpr;
    nSTATS.roc_prec = roc_prec;
    nSTATS.roc_rec = roc_rec;
    nSTATS.fsc = fsc;
    nSTATS.auc = auc;
    nSTATS.mcc = mcc;
end

%% END