%% Machine Learning ToolBox

% Online and Sequential Algorithms
% Author: David Nascimento Coelho
% Last Update: 2020/04/08

%% CHOOSE ALGORITHM

class_name = 'k2nn';
class_train = @k2nn_train;
class_test = @k2nn_classify;

%% DATA LOADING AND PRE-PROCESSING

% Load Dataset and Adjust its Labels

DATA = data_class_loading(OPT);     % Load Data Set
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

[Nc,N] = size(DATA.output);        	% get number of classes and samples

% Set data for the cross validation step
% min (0.2 * N, 1000)

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

% Normalize hpo data
DATAhpo = normalize(DATAhpo,OPT);

% Normalize ttt data
DATAttt.Xmax = DATAhpo.Xmax;
DATAttt.Xmin = DATAhpo.Xmin;
DATAttt.Xmed = DATAhpo.Xmed;
DATAttt.Xdp = DATAhpo.Xdp;
DATAttt = normalize(DATAttt,OPT);

%% DATA VISUALIZATION

figure; plot_data_pairplot(DATAttt);

%% ACCUMULATORS

accuracy_vector = zeros(1,Nttt);       % Hold Acc / (Acc + Err)
no_of_correct = zeros(1,Nttt);         % Hold # of correctly classified x
no_of_errors = zeros(1,Nttt);          % Hold # of misclassified x
predict_vector = zeros(2,Nttt);        % Hold true and predicted labels
no_of_samples = zeros(Nc,Nttt);        % Hold number of samples per class

figure; VID = struct('cdata',cell(1,Nttt),'colormap', cell(1,Nttt));

%% CROSS VALIDATION FOR HYPERPARAMETERS OPTIMIZATION

display('begin grid search')

HPo = grid_search_ttt(DATAhpo,HP_gs,class_train,class_test);

%% ADD FIRST ELEMENT TO DICTIONARY

% Get statistics from data (For Video Function)
DATAn.Xmax = max(DATAttt.input,[],2);	% max value
DATAn.Xmin = min(DATAttt.input,[],2);	% min value
DATAn.Xmed = mean(DATAttt.input,2);     % mean value
DATAn.Xdp = std(DATAttt.input,[],2);	% std value

% Add first element to dictionary
DATAn.input = DATA.input(:,1);      % First element input
DATAn.output = DATA.output(:,1);    % First element output
[~,max_y] = max(DATAn.output);      % Get sample's class
no_of_samples(max_y,1) = 1;         % Update number of samples per class
PAR = k2nn_train(DATAn,HPo);     	% Add element

% Update Video Function
if (HPo.Von),
    VID(1) = prototypes_frame(PAR.Cx,DATAn);
end

display('begin Test-than-train')

%% PRESEQUENTIAL (TEST-THAN-TRAIN)

for n = 2:Nttt,
    
    % Display number of samples already seen (for debug)
    
    if(mod(n,1000) == 0),
        disp(n);
        disp(datestr(now));
    end
    
    % Get current data
    
    DATAn.input = DATA.input(:,n);
    DATAn.output = DATA.output(:,n);
    
    [~,max_y] = max(DATAn.output);
    predict_vector(1,n) = max_y;
    
    for c = 1:Nc,
        if (c == max_y),
            no_of_samples(c,n) = no_of_samples(c,n-1) + 1;
        else
            no_of_samples(c,n) = no_of_samples(c,n-1);
        end
    end
    
    % Test (classify arriving data with current model)
    
    OUTn = k2nn_classify(DATAn,PAR);

    [~,max_yh] = max(OUTn.y_h);
    predict_vector(2,n) = max_yh;
    
    % Statistics
    
    if (max_y == max_yh),
        no_of_correct(n) = no_of_correct(n-1) + 1;
        no_of_errors(n) = no_of_errors(n-1);
    else
        no_of_correct(n) = no_of_correct(n-1);
        no_of_errors(n) = no_of_errors(n-1) + 1;
    end
    accuracy_vector(n) = no_of_correct(n) / ...
                        (no_of_correct(n) + no_of_errors(n));
    
    % Update score (for prunning method)
    
    PAR = k2nn_score_updt(DATAn,PAR,OUTn);
    
    % Train (with arriving data)
    
    PAR = k2nn_train(DATAn,PAR);
    
    % Video Function
    
    if (HP.Von),
        VID(n) = prototypes_frame(PAR.Cx,DATAn);
    end
    
end

%% SAVE FILE

save(OPT.file,'-v7.3')

%% END