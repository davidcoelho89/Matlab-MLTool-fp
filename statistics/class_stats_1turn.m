function [STATS] = class_stats_1turn(DATA,OUT,BIN)

% --- Provide Statistics of 1 turn of Classification ---
%
%   [STATS] = class_stats_1turn(DATA,OUT,bin)
% 
%   Input:
%    	DATA.
%           output = actual labels              [1 x N] or [Nc x N]
%     	OUT.
%           y_h = estimated labels              [1 x N] or [Nc x N]
%       BIN.
%           class1 = +1 binary class            [1 x Nc1]
%           class2 = -1 binary class            [1 x Nc2]
%   Output:
%       STATS.
%           Y = actual labels                   [1 x N] or [Nc x N]
%           Yh = estimated labels               [1 x N] or [Nc x N]
%       	Mconf = confusion matrix            [Nc x Nc]
%           Mconfs = one vs all Mconf           {1 x Nc} [2 x 2] (cell)
%           acc = % of accuracy                 [cte]
%           err = 1 - % accuracy           	    [cte]
%           roc_t = threshold                   [1 x len]
%           roc_prec = precision                [Nc x len]
%           roc_rec = recall                    [Nc x len]
%             (a.k.a. TPR or Sensitivity)
%           roc_spec = specificity              [Nc x len]
%             (a.k.a. TNR)
%           roc_fpr = false positive rate       [Nc x len]
%           auc = area under the curve          [Nc x 1]
%           fsc_per_class = f1-score            [Nc x 1]
%           fsc_micro = fsc micro avg           [cte]
%           fsc_macro = fsc macro avg           [cte]
%           fsc_weighted = fsc weighted avg     [cte]
%           mcc_per_class = Matthews Corr Coef  [Nc x 1]
%           mcc_mean                            [cte]
%           mcc_multiclass                      [cte]

%% INITIALIZATIONS

% Estimated and Actual Labels 
Y = DATA.output;
Yh = OUT.y_h;

% Get number of samples and classes
[Nc,N] = size(Y);

% Convert Problem into Binary Problem
if (nargin == 3)
    Y_aux = -1*ones(2,N);
    Yh_aux = -1*ones(2,N);
    class1 = BIN.class1;
    class2 = BIN.class2;
    [~,Y_seq] = max(Y);
    [~,Yh_seq] = max(Yh);
    for i = 1:N
        % Binary Class for Actual Labels
        if (any(class1 == Y_seq))
            Y_aux(1,i) = 1;
        elseif(any(class2 == Y_seq))
            Y_aux(2,i) = 1;
        end
        % Binary Class for Estimated Labels
        if(any(class1 == Yh_seq))
            Yh_aux(1,i) = 1;
        elseif(any(class2 == Yh_seq))
            Yh_aux(2,i) = 1;
        end
    end
    % Update Estimated and Actual Labels
    Y = Y_aux;
    Yh = Yh_aux;
    Nc = 2;
end

if Nc == 1
    Nc = length(unique(Y));
    classType = 1;  % ordinal
else
    classType = 2;  % [0 1] or [-1 +1]
end

%% Calculate Confusion Matrix

Mconf = zeros(Nc,Nc);

Mconfs = cell(Nc,1);
for i = 1:Nc
    Mconfs{i} = zeros(2,2);
end

samples_per_class = zeros(Nc,1);

for i = 1:N
    
    y_i = Y(:,i);       % get actual label
    yh_i = Yh(:,i);     % get estimated label
    
    if classType == 1
        % Ordinal - ToDo

    elseif classType == 2

        % [0 1] or [-1 +1]
        [~,iY] = max(y_i);
        [~,iY_h] = max(yh_i);

        % Get number of samples per class
        samples_per_class(iY) = samples_per_class(iY) + 1;

        % Get multiclass Confusion Matrix
        Mconf(iY_h,iY) = Mconf(iY_h,iY) + 1;
        
        % Get one-vs-all binary confusion matrices
        for j = 1:Nc

            if iY == j
                Y_pos_c = 1;
            else
                Y_pos_c = 2;
            end

            if iY_h == j
                Yh_pos_c = 1;
            else
                Yh_pos_c = 2;
            end

            Mconfs{j}(Yh_pos_c,Y_pos_c) = Mconfs{j}(Yh_pos_c,Y_pos_c) + 1;
            
        end

    end
end

%% Calculate Accuracy and Error

correct = trace(Mconf);
acc = correct/N;
err = 1 - acc;

%% Calculate F1-score and MCC (Matthews Correlation Coefficient)

% References

% F1-score
% (can also calculte Fbeta-score)
% https://www.v7labs.com/blog/f1-score-guide

% Matthews Correlation Coefficient (MCC)
% (Rk and MCC are the same for confusion matrices)
% https://dwbi1.wordpress.com/2022/10/05/mcc-formula-for-multiclass-classification/
% https://github.com/tensorflow/addons/issues/2339
% https://blester125.com/blog/rk.html

% Calculate class-wise (F1s and MCC)

f1_score_per_class = zeros(Nc,1);
mcc_per_class = zeros(Nc,1);

tp_bin = zeros(Nc,1);
fp_bin = zeros(Nc,1);
tn_bin = zeros(Nc,1);
fn_bin = zeros(Nc,1);

prec_bin = zeros(Nc,1);
rec_bin = zeros(Nc,1);
spec_bin = zeros(Nc,1);
fpr_bin = zeros(Nc,1);

for c = 1:Nc
    
    % TP, FP, TN, FN
    tp_bin(c) = Mconfs{c}(1,1);
    fp_bin(c) = Mconfs{c}(1,2);
    tn_bin(c) = Mconfs{c}(2,2);
    fn_bin(c) = Mconfs{c}(2,1);

    % PRECISION, RECALL
    if tp_bin(c) == 0
        prec_bin(c) = 0;
        rec_bin(c) = 0;
    else
        prec_bin(c) = tp_bin(c) / (tp_bin(c) + fp_bin(c));
        rec_bin(c) = tp_bin(c) / (tp_bin(c) + fn_bin(c));
    end
    
    % SPECIFICITY, FPR
    if tn_bin(c) == 0
        spec_bin(c) = 0;
        fpr_bin(c) = 1;
    else
        spec_bin(c) = tn_bin(c) / (tn_bin(c) + fp_bin(c));
        fpr_bin(c) = 1 - spec_bin(c);
    end

    % F1-SCORE binary

    num_f1_score_bin = 2 * prec_bin(c) * rec_bin(c);
    den_f1_score_bin = prec_bin(c) + rec_bin(c);

    if den_f1_score_bin == 0
        f1_score_per_class(c) = 0;
    else
        f1_score_per_class(c) = num_f1_score_bin / den_f1_score_bin;
    end
    
    % MCC binary

    sum1 = tp_bin(c) + fp_bin(c);
    sum2 = tp_bin(c) + fn_bin(c);
    sum3 = tn_bin(c) + fp_bin(c);
    sum4 = tn_bin(c) + fn_bin(c);
    if (sum1 == 0 || sum2 == 0 || sum3 == 0 || sum4 == 0)
        den_mcc = 1;
    else
        den_mcc = (sum1*sum2*sum3*sum4)^0.5;
    end
    
    mcc_per_class(c) = (tp_bin(c)*tn_bin(c) - fp_bin(c)*fn_bin(c))/den_mcc;

end

% F1-SCORE multiclass

f1_score_macro = sum(f1_score_per_class)/Nc;

f1_score_micro = sum(tp_bin) / ...
                 ( sum(tp_bin) + 0.5 * (sum(fp_bin) + sum(fn_bin)) );

f1_score_weighted = 0;
for c = 1:Nc
    wc = samples_per_class(c) / N;
    f1_score_weighted = f1_score_weighted + wc * f1_score_per_class(c);
end

% MCC multiclass

mcc_mean = sum(mcc_per_class)/Nc;

pk_vector = sum(Mconf,1);
tk_vector = sum(Mconf,2);

cov_tk_pk = correct * N - pk_vector*tk_vector;
cov_pk_pk = N * N - pk_vector*pk_vector';
cov_tk_tk = N * N - tk_vector'*tk_vector;

num_mcc = cov_tk_pk;

den_mcc = ( cov_pk_pk * cov_tk_tk) ^ 0.5;
if(den_mcc == 0)
    den_mcc = 1;
end

mcc_multiclass = num_mcc / den_mcc;

%% Calculate ROC (Receiver Operating Characteristic) Curve

% Hyperparameters for ROC Curve

disc = 0.1;         % Discretization
len = (2/disc)+1;   % length of roc curve vectors

ROC_t = -1:disc:1;  % Threshold vector

% Ordinal Classification - Todo

if classType == 1

    ROC_PREC = [];      % Precision (positive predictive value)
    ROC_REC = [];       % Recall (used for unbalanced data)
    ROC_SPEC = [];      % Specificity
    ROC_FPR = [];       % False Positive Rate (1 - specificity)

% [0 1] or [-1 +1] Classification

else

    % Init outputs

    ROC_PREC = zeros(Nc,len);
    ROC_REC = zeros(Nc,len);
    ROC_SPEC = zeros(Nc,len);
    ROC_FPR = zeros(Nc,len);

    % One ROC curve for each class

    for c = 1:Nc

        y = Y(c,:);         % actual labels for class c
        yh = Yh(c,:);       % predicted labels for class c

        cont = 0;           % Counter

        for t = -1:disc:1   % t limiar

            cont = cont + 1;
            Mconf_roc = zeros(2,2);

            for i = 1:N    % calculate confusion matrix

                % Get actual and predicted labels
                y_i = y(i);
                yh_i = yh(i);

                % Get actual label position for confusion matrix
                if y_i == 1
                    y_pos = 1;
                else
                    y_pos = 2;
                end

                % Get predicted label position for confusion matrix
                if (yh_i > t)
                    yh_pos = 1;
                else
                    yh_pos = 2;
                end

                % Update confusion matrix
                Mconf_roc(yh_pos,y_pos) = Mconf_roc(yh_pos,y_pos) + 1;

            end

            % Get True and False, Positives and Negatives.
            TP = Mconf_roc(1,1);
            TN = Mconf_roc(2,2);
            FP = Mconf_roc(1,2);
            FN = Mconf_roc(2,1);

            % Get ROC curve vectors
            ROC_PREC(c,cont) = TP / (TP + FP);
            ROC_REC(c,cont)  = TP / (TP + FN);
            ROC_SPEC(c,cont) = TN / (TN + FP);
            ROC_FPR(c,cont)  = 1 - (TN / (TN + FP));

        end

    end

end

%% Calculate AUC

% Needs "another loop" for multiclass 

AUC = zeros(Nc,1);

for c = 1:Nc
    AUC(c) = sum(ROC_REC(c,:))/len;
end

%% FILL OUTPUT STRUCTURE

STATS.Y = Y;
STATS.Yh = Yh;
STATS.Mconf = Mconf;
STATS.Mconfs = Mconfs;
STATS.acc = acc;
STATS.err = err;

STATS.fsc_per_class = f1_score_per_class;
STATS.fsc_micro = f1_score_micro;
STATS.fsc_macro = f1_score_macro;
STATS.fsc_weighted = f1_score_weighted;

STATS.mcc_per_class = mcc_per_class;
STATS.mcc_mean = mcc_mean;
STATS.mcc_multiclass = mcc_multiclass;

STATS.roc_t = ROC_t;
STATS.roc_prec = ROC_PREC;
STATS.roc_rec = ROC_REC;
STATS.roc_spec = ROC_SPEC;
STATS.roc_fpr = ROC_FPR;
STATS.auc = AUC;

%% END