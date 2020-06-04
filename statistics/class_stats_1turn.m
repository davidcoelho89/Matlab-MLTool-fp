function [STATS] = class_stats_1turn(DATA,OUT,BIN)

% --- Provide Statistics of 1 turn of Classification ---
%
%   [STATS] = class_stats_1turn(DATA,OUT,bin)
% 
%   Input:
%    	DATA.
%           output = actual labels          [1 x N] or [Nc x N]
%     	OUT.
%           y_h = estimated labels          [1 x N] or [Nc x N]
%       BIN.
%           class1 = +1 binary class        [1 x Nc1]
%           class2 = -1 binary class        [1 x Nc2]
%   Output:
%       STATS.
%           Y = actual labels               [1 x N] or [Nc x N]
%           Yh = estimated labels           [1 x N] or [Nc x N]
%       	Mconf = confusion matrix        [Nc x Nc]
%           acc = % of accuracy             [cte]
%           err = 1 - % accuracy           	[cte]
%           roc_t = threshold               [1 x len]
%           roc_tpr = true positive rate    [Nc x len] (sensitivity)
%           roc_spec = specificity          [Nc x len]
%           roc_fpr = false positive rate   [Nc x len] (1 - specificity)
%           roc_prec = precision            [Nc x len]
%           roc_rec = recall                [Nc x len]
%           fsc = f1-score                  [Nc x 1]
%           auc = area under the curve      [Nc x 1]
%           mcc = Matthews Correlation Coef [Nc x 1]

%% INITIALIZATIONS

% Estimated and Actual Labels 
Y = DATA.output;
Yh = OUT.y_h;

% Get number of samples and classes
[Nc,N] = size(Y);

% Convert Problem into Binary Problem
if (nargin == 3),
    Y_aux = -1*ones(2,N);
    Yh_aux = -1*ones(2,N);
    class1 = BIN.class1;
    class2 = BIN.class2;
    [~,Y_seq] = max(Y);
    [~,Yh_seq] = max(Yh);
    for i = 1:N,
        % Binary Class for Actual Labels
        if (any(class1 == Y_seq)),
            Y_aux(1,i) = 1;
        elseif(any(class2 == Y_seq))
            Y_aux(2,i) = 1;
        end
        % Binary Class for Estimated Labels
        if(any(class1 == Yh_seq)),
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

if Nc == 1, 
    Nc = length(unique(Y));
    classType = 1;  % ordinal
else
    classType = 2;  % [0 1] or [-1 +1]
end

%% Calculate Confusion Matrix

Mconf = zeros(Nc,Nc);

for i = 1:N,
    
    y_i = Y(:,i);       % get actual label
    yh_i = Yh(:,i);     % get estimated label
    
    if classType == 1,
        % Ordinal - ToDo
    elseif classType == 2,
        % [0 1] or [-1 +1]
        [~,iY] = max(y_i);
        [~,iY_h] = max(yh_i);
        Mconf(iY,iY_h) = Mconf(iY,iY_h) + 1;
    end
end

%% Calculate Accuracy and Error

acc = trace(Mconf)/N;
err = 1 - acc;

%% Calculate ROC (Receiver Operating Characteristic) Curve

% Hyperparameters for ROC Curve

disc = 0.1;         % Discretization
len = (2/disc)+1;   % length of roc curve vectors

ROC_t = -1:disc:1;  % Threshold vector

% Ordinal Classification - Todo

if classType == 1,
    
    ROC_TPR = [];       % True Positive Rate (sensitivity)
    ROC_SPEC = [];      % Specificity
    ROC_FPR = [];       % False Positive Rate (1 - specificity)
    ROC_PREC = [];      % Precision (positive predictive value)
    ROC_REC = [];       % Recall (used for unbalanced data)
    FSC = [];           % F-score
    MCC = [];           % Matthews Correlation Coefficient

% [0 1] or [-1 +1] Classification
    
else
    
    % Init outputs
    
    ROC_TPR = zeros(Nc,len);
    ROC_SPEC = zeros(Nc,len);
    ROC_FPR = zeros(Nc,len);
    ROC_PREC = zeros(Nc,len);
    ROC_REC = zeros(Nc,len);
    FSC = zeros(Nc,1);
    MCC = zeros(Nc,1);
    
    % One ROC curve for each classifier
    
    for c = 1:Nc,
        
        y = Y(c,:);         % actual labels for class c
        yh = Yh(c,:);       % predicted labels for class c
        
        cont = 0;           % Counter
        
        for t = -1:disc:1,   % t limiar
            
            cont = cont + 1;
            Mconf_roc = zeros(2,2);
            
            for i = 1:N,    % calculate confusion matrix
                
                % Get actual and predicted labels
                y_i = y(i);
                yh_i = yh(i);
                
                % Get actual label position for confusion matrix
                if y_i == 1,
                    y_pos = 1;
                else
                    y_pos = 2;
                end
                
                % Get predicted label position for confusion matrix
                if (yh_i > t),
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
            ROC_TPR(c,cont)  = TP / (TP + FN);
            ROC_SPEC(c,cont) = TN / (TN + FP);
            ROC_FPR(c,cont)  = 1 - (TN / (TN + FP));
            ROC_PREC(c,cont) = TP / (TP + FP);
            ROC_REC(c,cont)  = TP / (TP + FN);
            
            if (t == 0),
                % Get f1-score
                FSC(c) = 2 * ROC_PREC(c,cont) * ROC_REC(c,cont) / ...
                        ( ROC_PREC(c,cont) + ROC_REC(c,cont) );
                % Get Matthews Correlation Coefficient
                sum1 = TP + FP; sum2 = TP + FN;
                sum3 = TN + FP; sum4 = TN + FN;
                if (sum1 == 0 || sum2 == 0 || sum3 == 0 || sum4 == 0),
                    den = 1;
                else
                    den = (sum1*sum2*sum3*sum4)^0.5;
                end
                MCC(c) = (TP*TN - FP*FN)/den;
            end
            
        end
        
    end
    
end

%% Calculate AUC

% Needs "another loop" for multiclass 

AUC = zeros(Nc,1);

for c = 1:Nc,
    AUC(c) = sum(ROC_TPR(c,:))/len;
end

%% FILL OUTPUT STRUCTURE

STATS.Y = Y;
STATS.Yh = Yh;
STATS.Mconf = Mconf;
STATS.acc = acc;
STATS.err = err;
STATS.roc_t = ROC_t;
STATS.roc_tpr = ROC_TPR;
STATS.roc_spec = ROC_SPEC;
STATS.roc_fpr = ROC_FPR;
STATS.roc_prec = ROC_PREC;
STATS.roc_rec = ROC_REC;
STATS.fsc = FSC;
STATS.auc = AUC;
STATS.mcc = MCC;

%% END