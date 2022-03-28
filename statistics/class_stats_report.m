function [] = class_stats_report(STATS)

% --- Provide the report of a classification experiment ---
%
%   [] = class_stats_report(STATS)
% 
%   Input:
%    	STATS = Cell containing statistics of various classifiers
%   Output:
%       "void" (print graphics at screen, and info at command window)

%% INITIALIZATIONS

Nr = length(STATS.acc);             % Number of realizations
Nc = length(STATS.fsc{1,1});        % Number of classes

Mat_boxplot_sens = zeros(Nr,Nc);    % Boxplot matrix of sensitivity
Mat_boxplot_spec = zeros(Nr,Nc);    % Boxplot matrix of specificity
% Mat_boxplot_gm = zeros(Nr,Nc);      % Boxplot matrix of Geom Mean
Mat_boxplot_f1s = zeros(Nr,Nc);     % Boxplot matrix of F1-score
Mat_boxplot_auc = zeros(Nr,Nc);     % Boxplot matrix of Area Under the Curve
Mat_boxplot_mcc = zeros(Nr,Nc);     % Boxplot matrix of Matthews Correlation Coef

%% ALGORITHM - ERROR AND ACCURACY

% Find index of "0" bias
roc_t = STATS.roc_t{1,1};
t0_index = 0;
for i = 1:length(roc_t)
    if (roc_t(i) == 0)
        t0_index = i;
    end
end

% Box plot of Accuracy

figure; boxplot(STATS.acc);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Accuracy')
% xlabel('Classifiers')
title('Results')
axis ([0 2 0 1.05])

disp('Accuracy: Max, Min, Mean, Median, Std:')
disp(STATS.acc_max);
disp(STATS.acc_min);
disp(STATS.acc_mean);
disp(STATS.acc_median);
disp(STATS.acc_std);

% Box plot of Error

figure; boxplot(STATS.err);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Error')
% xlabel('Classifiers')
title('Results')
axis ([0 2 0 1.05])

% Display information at command line

disp('Error: Max, Min, Mean, Median, Std:')
disp(STATS.err_max);
disp(STATS.err_min);
disp(STATS.err_mean);
disp(STATS.err_median);
disp(STATS.err_std);

%% ALGORITHM - Sensitivity, Specificity, GM, F1score, AUC, MCC

% Box plot of Sensitivity (for each class)
% (a.k.a. TPR or Recall)

for i = 1:Nr
    sens = STATS.roc_rec{1,i};
    Mat_boxplot_sens(i,:) = sens(:,t0_index);
end

figure; boxplot(Mat_boxplot_sens);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Sensitivity')
xlabel('Class')
title('Classification Results')
axis ([0 Nc+1 -0.05 1.05])

hold on
media_sens = mean(Mat_boxplot_sens);  	% Mean Sensitivity
median_sens = median(Mat_boxplot_sens);	% Median Sensitivity

plot(media_sens,'*k')
hold off

% Box plot of Specificity (for each class)

for i = 1:Nr
    spec = STATS.roc_spec{1,i};
    Mat_boxplot_spec(i,:) = spec(:,t0_index);
end

figure; boxplot(Mat_boxplot_spec);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Specificity')
xlabel('Class')
title('Classification Results')
axis ([0 Nc+1 -0.05 1.05])

hold on
media_spec = mean(Mat_boxplot_spec);    % Mean Specificity
median_spec = median(Mat_boxplot_spec);	% Median Specificity
plot(media_spec,'*k')
hold off

% Box plot of GM = sqrt(SENS*ESPC) (for each class)

Mat_boxplot_gm = sqrt(Mat_boxplot_sens.*Mat_boxplot_spec);

figure; boxplot(Mat_boxplot_gm);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('GM = sqrt(SENS*SPEC)')
xlabel('Class')
title('Classification Results')
axis ([0 Nc+1 -0.05 1.05])

hold on
media_gm = mean(Mat_boxplot_gm);    % Mean Geometric mean
median_gm = median(Mat_boxplot_gm);	% Median Geometric mean
plot(media_gm,'*k')
hold off

% Box plot of f1score (for each class)

for i = 1:Nr
    fsc = STATS.fsc{1,i};
    Mat_boxplot_f1s(i,:) = fsc';
end

figure; boxplot(Mat_boxplot_f1s);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('F1-Score')
xlabel('Class')
title('Classification Results')
axis ([0 Nc+1 -0.05 1.05])

hold on
media_fsc = mean(Mat_boxplot_f1s);      % Mean F1-score
median_fsc = median(Mat_boxplot_f1s);   % Median F1-score
plot(media_fsc,'*k')
hold off

% Box plot of AUC (for each class)

for i = 1:Nr
    auc = STATS.auc{1,i};
    Mat_boxplot_auc(i,:) = auc';
end

figure; boxplot(Mat_boxplot_auc);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Area Under the Curve')
xlabel('Class')
title('Classification Results')
axis ([0 Nc+1 -0.05 1.05])

hold on
media_auc = mean(Mat_boxplot_auc);      % Mean Area Under the Curve
median_auc = median(Mat_boxplot_auc);   % Median Area Under the Curve

plot(media_auc,'*k')
hold off

% Box plot of MCC (for each class)

for i = 1:Nr
    mcc = STATS.mcc{1,i};
    Mat_boxplot_mcc(i,:) = mcc';
end

figure; boxplot(Mat_boxplot_mcc);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Matthews Correlation Coefficient')
xlabel('Class')
title('Classification Results')
axis ([0 Nc+1 -0.05 1.05])

hold on
media_mcc = mean(Mat_boxplot_mcc);      % Mean Matthews Correlation Coef
median_mcc = median(Mat_boxplot_mcc);   % Median Matthews Correlation Coef
plot(media_mcc,'*k')
hold off

% Display information at command line

disp('Mean - Sensitivity, Specificity, GM, F1score, AUC, MCC');
disp(median_sens);
disp(median_spec);
disp(median_gm);
disp(median_fsc);
disp(median_auc);
disp(median_mcc);

%% END