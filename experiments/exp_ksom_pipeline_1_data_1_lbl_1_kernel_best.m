function [] = exp_ksom_pipeline_1_data_1_lbl_1_kernel_best(DATA,OPT)

% --- Pipeline used to test ksom model with 1 dataset and 1 Kernel ---
%
%   [] = exp_ksom_pipeline_1_data_1_lbl_1_kernel_best(DATA,OPT)
%   Input:
%       DATA.
%           input = attributes matrix                   [p x N]
%           output = labels matrix                      [Nc x N]
%       OPT.
%           prob = which dataset will be used
%           prob2 = a specification of the dataset
%           norm = which normalization will be used
%           lbl = which labeling strategy will be used
%   Output:
%       "Do not have. Just save structures into a file"

%% DATA PRE-PROCESSING AND VISUALIZATION

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

% plot_data_pairplot(DATA);           % See pairplot of attributes

% [Nc,~] = size(DATA.output);         % Get number of classes

%% ACCUMULATORS

NAMES = {'train','test'};               % Names for plots

% data_acc = cell(OPT.Nr,1);              % Acc of labels and data division

nstats_all = cell(length(NAMES),1);     % Group Stats from Tr and Ts

par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM
out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

%% HANDLERS FOR CLASSIFICATION FUNCTIONS

str_train = strcat(lower(OPT.alg),'_train');
class_train = str2func(str_train);

str_test = strcat(lower(OPT.alg),'_classify');
class_test = str2func(str_test);

%% GET HP FROM FILE

variables = load(OPT.file_hp);
max_acc_i = variables.nstats_all{2,1}.acc_max_i;

if(isfield(variables,'ksomgd_par_acc'))
    get_par_acc = variables.ksomgd_par_acc;
elseif(isfield(variables,'ksomef_par_acc'))
    get_par_acc = variables.ksomef_par_acc;
elseif(isfield(variables,'par_acc'))
    get_par_acc = variables.par_acc;
end

PAR = get_par_acc{max_acc_i,1};

HP = struct();
HP.Nep = PAR.Nep;
HP.Nk = PAR.Nk;
HP.init = PAR.init;
HP.dist = PAR.dist;
HP.learn = PAR.learn;
HP.No = PAR.No;
HP.Nt = PAR.Nt;
HP.Nn = PAR.Nn;
HP.neig = PAR.neig;
HP.Vo = PAR.Vo;
HP.Vt = PAR.Vt;
HP.lbl = PAR.lbl;
HP.Von = PAR.Von;
HP.Ktype = PAR.Ktype;

if(isfield(PAR,'sigma'))
    HP.sigma = PAR.sigma;
end
if(isfield(PAR,'alpha'))
    HP.alpha = PAR.alpha;
end
if(isfield(PAR,'theta'))
    HP.theta = PAR.theta;
end
if(isfield(PAR,'gamma'))
    HP.gamma = PAR.gamma;
end

%% HOLD OUT / NORMALIZE / SHUFFLE / TRAINING / TEST / STATISTICS

for r = 1:OPT.Nr
   
% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp(r);
display(datetime("now"));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

[DATAho] = hold_out(DATA,OPT);

% data_acc{r} = DATAho;
DATAtr = DATAho.DATAtr;
DATAts = DATAho.DATAts;

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

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

par_acc{r} = class_train(DATAtr,HP);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

out_tr_acc{r} = class_test(DATAtr,par_acc{r});
stats_tr_acc{r} = class_stats_1turn(DATAtr,out_tr_acc{r});

out_ts_acc{r} = class_test(DATAts,par_acc{r});
stats_ts_acc{r} = class_stats_1turn(DATAts,out_ts_acc{r});

end

%% RESULTS / STATISTICS

% Statistics for n turns (multiclass)

nstats_tr = class_stats_nturns(stats_tr_acc);
nstats_ts = class_stats_nturns(stats_ts_acc);

% Statistics for n turns (binary)

% if(OPT.calculate_bin == 1)
% 
% end

% Get all Statistics in one Cell

nstats_all{1,1} = nstats_tr;
nstats_all{2,1} = nstats_ts;

% Compare Training and Test Statistics

class_stats_ncomp(nstats_all,NAMES);

%% SAVE DATA

if(OPT.savefile)
    save(OPT.file);
end

%% END