function [] = exp_knns_pipeline_streaming_1data(OPT)

% --- Pipeline used to test knns model with 1 dataset ---
%   [] = exp_knns_pipeline_streaming_1data(OPT)
%
%   Input:
%       OPT.
%           prob = which dataset will be used
%           prob2 = a specification of the dataset
%           norm = which normalization will be used
%           lbl = which labeling strategy will be used
%   Output:
%       "Do not have. Just save structures into a file"

%% DATA LOADING

DATA = data_class_loading(OPT);

%% HYPERPARAMETERS - DEFAULT

HP_gs.dist = 2;        % Type of distance = euclidean
HP_gs.Ws = 100;        % Window size
HP_gs.Ktype = 0;       % Non-kernelized Algorithm
HP_gs.K = 5;           % Number of nearest neighbors
HP_gs.knn_type = 1;    % Majority voting KNN
HP_gs.Von = 0;         % Disable Video

% Dont have hyperparameter optimization

PAR = HP_gs;

%% FILE NAME - STRINGS

str1 = DATA.name;
str2 = '_knns_norm';
str3 = int2str(OPT.norm);
str4 = '_windowSize';
str5 = int2str(HP_gs.Ws);
str6 = '_';
str7 =int2str(HP_gs.K);
str8 = 'nn.mat';

OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8);

disp(OPT.file);

%% DATA PRE-PROCESSING 

% Adjust Dataset Labels and Get its Dimensions

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

[Nc,N] = size(DATA.output);        	% get number of classes and samples

% Test-than-train data

Nttt = N;
DATAttt.input = DATA.input;
DATAttt.output = DATA.output;

%% DATA NORMALIZATION

% Get Normalization Parameters
PARnorm = normalize_fit(DATAttt,OPT);

% Normalize all data
DATA = normalize_transform(DATA,PARnorm);

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

%% PRESEQUENTIAL (TEST-THAN-TRAIN)

disp('begin Test-than-train')

for n = 1:Nttt
    
    % Display number of samples already seen (for debug)
    
    if(mod(n,1000) == 0)
        disp(n);
        disp(datestr(now));
    end
    
    % Get current data
    
    DATAn.input = DATAttt.input(:,n);
    DATAn.output = DATAttt.output(:,n);
    [~,y_lbl] = max(DATAn.output);
    
    % Test  (classify arriving data with current model)
    % Train (update model with arriving data)
    
    PAR = knns_train(DATAn,PAR);
    
    % Hold Number of Samples per Class 
    
    if n == 1
        samples_per_class(y_lbl,n) = 1; % first element
    else
        samples_per_class(:,n) = samples_per_class(:,n-1);
        samples_per_class(y_lbl,n) = samples_per_class(y_lbl,n-1) + 1;
    end
    
    % Hold Predicted Labels
    
    predict_vector(:,n) = PAR.y_h;
    [~,yh_lbl] = max(PAR.y_h);
    
    % Hold Number of Errors and Hits
    
    if n == 1
        if (y_lbl == yh_lbl)
            no_of_correct(n) = 1;
        else
            no_of_errors(n) = 1;
        end
    else
        if (y_lbl == yh_lbl)
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
    for c = 1:Nc
        prot_per_class(c,n) = sum(lbls == c);
    end
    
    [~,Nprot] = size(PAR.Cy);
    prot_per_class(Nc+1,n) = Nprot;
    
    % Video Function
    
    if (PAR.Von)
        VID(n) = prototypes_frame(PAR.Cx,DATAn);
    end
    
end

%% SAVE FILE

save(OPT.file,'-v7.3')

%% END