function [] = exp_spok_stationary_pipeline_1data_1Ss_1kernel(DATA,...
                                                             OPT,...
                                                             HPgs,...
                                                             CVp)

% --- Pipeline used to test spok model with 1 dataset and 1 Kernel ---
%
%   [] = exp_spok_stationary_pipeline_1data_1Ss_1kernel(DATA,OPT,HPgs,PSp)
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
%       CVp.
%           fold = % number of data partitions for cross validation
%                        presented to the algorithm
%           cost = type of cross validation                         [cte]
%               1: takes into account just accurary
%               2: takes into account also the dicitionary size
%           lambda = trade-off between error and dictionary size    [0 - 1]
%   Output:
%       "Do not have. Just save structures into a file"

%% DATA PRE-PROCESSING AND VISUALIZATION

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

% plot_data_pairplot(DATA);           % See pairplot of attributes

%% ACCUMULATORS

NAMES = {'train','test'};           % Acc of names for plots

% data_acc = cell(OPT.Nr,1);       	% Acc of Data

nstats_all = cell(2,1);             % Acc of General statistics

par_acc = cell(OPT.Nr,1);         	% Acc of Parameters and Hyperparameters
stats_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
stats_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data

%% HOLD OUT / NORMALIZE / SHUFFLE / HPO / TRAINING / TEST / STATISTICS

for r = 1:OPT.Nr

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp(r);
display(datetime("now"));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

DATAho = hold_out(DATA,OPT);	% Hold Out Function

% data_acc{r} = DATAho;
DATAtr = DATAho.DATAtr;         % Training Data
DATAts = DATAho.DATAts;     	% Test Data

HPgs.max_prot = floor( ((CVp.fold-1) / CVp.fold) * size(DATAtr.input, 2) );

% %%%%%%%%%%%%%%%%% NORMALIZE DATA %%%%%%%%%%%%%%%%%%%%%%%

% Get Normalization Parameters

PARnorm = normalize_fit(DATAtr,OPT);

% Training data normalization

DATAtr = normalize_transform(DATAtr,PARnorm);

% Test data normalization

DATAts = normalize_transform(DATAts,PARnorm);

% Adjust Values for video function

DATA = normalize_transform(DATA,PARnorm);
DATAtr.Xmax = max(DATA.input,[],2);  % max value
DATAtr.Xmin = min(DATA.input,[],2);  % min value
DATAtr.Xmed = mean(DATA.input,2);    % mean value
DATAtr.Xdp = std(DATA.input,[],2);   % std value

% %%%%%%%%%%%%%% SHUFFLE TRAINING DATA %%%%%%%%%%%%%%%%%%%

I = randperm(size(DATAtr.input,2));
DATAtr.input = DATAtr.input(:,I);
DATAtr.output = DATAtr.output(:,I);
DATAtr.lbl = DATAtr.lbl(:,I);

% %%%%%%%%%%% HYPERPARAMETER OPTIMIZATION %%%%%%%%%%%%%%%%

% Using Random Search and Cross-Validation
HP = random_search_cv(DATAtr,HPgs,@spok_train,@spok_classify,CVp);

% %%%%%%%%%%%%%% CLASSIFIER'S TRAINING %%%%%%%%%%%%%%%%%%%

% Calculate model's parameters
par_acc{r} = spok_train(DATAtr,HP);

% %%%%%%%%% CLASSIFIER'S TEST AND STATISTICS %%%%%%%%%%%%%

% Results and Statistics with training data
OUTtr = spok_classify(DATAtr,par_acc{r});
stats_tr_acc{r} = class_stats_1turn(DATAtr,OUTtr);

% Results and Statistics with test data
OUTts = spok_classify(DATAts,par_acc{r});
stats_ts_acc{r} = class_stats_1turn(DATAts,OUTts);

end

%% RESULTS / STATISTICS

% Statistics for n turns

nstats_tr = class_stats_nturns(stats_tr_acc);
nstats_ts = class_stats_nturns(stats_ts_acc);

% Get all Statistics in one Cell

nstats_all{1,1} = nstats_tr;
nstats_all{2,1} = nstats_ts;

% Compare Training and Test Statistics

class_stats_ncomp(nstats_all,NAMES);

%% SAVE FILE

save(OPT.file,'-v7.3')

%% END