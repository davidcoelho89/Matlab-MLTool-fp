function [] = exp_isk2nn_pipeline_streaming_1data_1Ss_1kernel(DATA,OPT,HPgs,PSp)

% --- Pipeline used to test isk2nn model with 1 dataset and 1 Kernel ---
%
%   [] = exp_isk2nn_pipeline_streaming_1data_1Ss_1kernel(DATA,OPT,HPgs,PSp)
%
%   Input:
%       DATA.
%           input = attributes matrix                   [p x N]
%           output = labels matrix                      [Nc x N]
%       OPT.
%           prob = which dataset will be used
%           prob2 = a specification of the dataset
%           norm = which normalization will be used
%           lbl = which labeling strategy will be used
%       HPgs = hyperparameters for grid searh of classifier
%             (vectors containing values that will be tested)
%       PSp.
%           iterations = number of times the data is 
%                        presented to the algorithm
%           type = type of cross validation                         [cte]
%               1: takes into account just accurary
%               2: takes into account also the dicitionary size
%           lambda = trade-off between error and dictionary size    [0 - 1]
%   Output:
%       "Do not have. Just save structures into a file"

%% DATA PRE-PROCESSING AND HOLD OUT

% Adjust Dataset Labels and Get its Dimensions

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

[Nc,N] = size(DATA.output);        	% get number of classes and samples

% Set data for the HyperParameter Optimization step: min (0.2 * N, 1000)

if (N < 5000),
    Nhpo = floor(0.2 * N);
else
    Nhpo = 1000;
end

DATAhpo.input = DATA.input(:,1:Nhpo);
DATAhpo.output = DATA.output(:,1:Nhpo);

% Set remaining data for test-than-train step

Nttt = N - Nhpo;

DATAttt.input = DATA.input(:,Nhpo+1:end);
DATAttt.output = DATA.output(:,Nhpo+1:end);

%% DATA NORMALIZATION

% Get Normalization Parameters
PARnorm = normalize_fit(DATAhpo,OPT);

% Normalize all data
DATA = normalize_transform(DATA,PARnorm);

% Normalize hpo data
DATAhpo = normalize_transform(DATAhpo,PARnorm);

% Normalize ttt data
DATAttt = normalize_transform(DATAttt,PARnorm);

% Get statistics from data (For Video Function)
DATAn.Xmax = max(DATA.input,[],2);
DATAn.Xmin = min(DATA.input,[],2);
DATAn.Xmed = mean(DATA.input,2);
DATAn.Xstd = std(DATA.input,[],2);

%% DATA VISUALIZATION

% figure; plot_data_pairplot(DATAttt);

%% ACCUMULATORS

samples_per_class = zeros(Nc,Nttt);	% Hold number of samples per class

predict_vector = zeros(Nc,Nttt);	% Hold predicted labels

no_of_correct = zeros(1,Nttt);      % Hold # of correctly classified x
no_of_errors = zeros(1,Nttt);       % Hold # of misclassified x

accuracy_vector = zeros(1,Nttt);	% Hold Acc / (Acc + Err)

prot_per_class = zeros(Nc+1,Nttt);	% Hold number of prot per class
                                    % Last is for the sum
                                    
VID = struct('cdata',cell(1,Nttt),'colormap', cell(1,Nttt));

%% CROSS VALIDATION FOR HYPERPARAMETERS OPTIMIZATION

display('begin grid search')
disp(datestr(now));

% Grid Search Parameters

if (nargin == 2),
    PSp.iterations = 1;
    PSp.type = 1;
    PSp.lambda = 0.5;
end

% Get Hyperparameters Optimized and the Prototypes Initialized

PAR = grid_search_ttt(DATAhpo,HPgs,@isk2nn_train,@isk2nn_classify,PSp);

% Change maximum number of prototypes

% PAR.max_prot = Inf;

%% PRESEQUENTIAL (TEST-THAN-TRAIN)

display('begin Test-than-train')

figure; % new figure for video ploting

for n = 1:Nttt,
    
    % Display number of samples already seen (for debug)
    
    if(mod(n,1000) == 0),
        disp(n);
        disp(datestr(now));
    end
    
    % Get current data
    
    DATAn.input = DATAttt.input(:,n);
    DATAn.output = DATAttt.output(:,n);
    [~,y_lbl] = max(DATAn.output);
    
    % Test  (classify arriving data with current model)
    % Train (update model with arriving data)
    
    PAR = isk2nn_train(DATAn,PAR);
    
    % Hold Number of Samples per Class 
    
    if n == 1,
        samples_per_class(y_lbl,n) = 1; % first element
    else
        samples_per_class(:,n) = samples_per_class(:,n-1);
        samples_per_class(y_lbl,n) = samples_per_class(y_lbl,n-1) + 1;
    end
    
    % Hold Predicted Labels
    
    predict_vector(:,n) = PAR.y_h;
    [~,yh_lbl] = max(PAR.y_h);
    
    % Hold Number of Errors and Hits
    
    if n == 1,
        if (y_lbl == yh_lbl),
            no_of_correct(n) = 1;
        else
            no_of_errors(n) = 1;
        end
    else
        if (y_lbl == yh_lbl),
            no_of_correct(n) = no_of_correct(n-1) + 1;
            no_of_errors(n) = no_of_errors(n-1);
        else
            no_of_correct(n) = no_of_correct(n-1);
            no_of_errors(n) = no_of_errors(n-1) + 1;
        end
    end
    
    % Hold Accuracy
    
    accuracy_vector(n) = no_of_correct(n) / ...
                        (no_of_correct(n) + no_of_errors(n));
    
    % Hold Number of prototypes per Class
    
    [~,lbls] = max(PAR.Cy);
    for c = 1:Nc,
        prot_per_class(c,n) = sum(lbls == c);
    end
    
    [~,Nprot] = size(PAR.Cy);
    prot_per_class(Nc+1,n) = Nprot;
    
    % Video Function
    
    if (PAR.Von),
        VID(n) = prototypes_frame(PAR.Cx,DATAn);
    end
    
end

%% SAVE FILE

save(OPT.file,'-v7.3')

%% END