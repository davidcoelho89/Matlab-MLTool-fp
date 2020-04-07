%% Machine Learning ToolBox

% Clustering Algorithms - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2019/11/12

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 11;              % Which problem will be solved / used
OPT.prob2 = 01;             % More details about a specific data set
OPT.norm = 3;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.Nr = 02;                % Number of repetitions of the algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

%% CHOOSE ALGORITHM

% Handlers for clustering functions

cluster_name = 'SOM';
cluster_alg = @som_cluster;
label_alg = @som_label;

%% CHOOSE HYPERPARAMETERS

HP.Nep = 100;     	% max number of epochs
HP.Nk = [3 3];    	% number of neurons (prototypes)
HP.init = 2;     	% neurons' initialization
HP.dist = 2;      	% type of distance
HP.learn = 2;     	% type of learning step
HP.No = 0.7;       	% initial learning step
HP.Nt = 0.01;      	% final learnin step
HP.Nn = 1;      	% number of neighbors
HP.neig = 3;      	% type of neighborhood function
HP.Vo = 0.8;      	% initial neighborhood constant
HP.Vt = 0.3;      	% final neighborhood constant
HP.lbl = 1;         % Neurons' labeling function
HP.Von = 1;         % enable/disable video 
HP.K = 1;           % Number of nearest neighbors
HP.Ktype = 0;       % Non-kernelized Algorithm

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set

DATA = normalize(DATA,OPT);         % normalize the attributes' matrix

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

DATA.Xmax = max(DATA.input,[],2);  % max value
DATA.Xmin = min(DATA.input,[],2);  % min value
DATA.Xmed = mean(DATA.input,2);    % mean value
DATA.Xdp = std(DATA.input,[],2);   % std value

%% ACCUMULATORS

PAR_acc = cell(OPT.Nr,1);   	% Acc of Parameters and Hyperparameters

STATS_acc = cell(OPT.Nr,1);   	% Acc of Statistics

%% CLUSTERING

display('Begin Algorithm');

for r = 1:OPT.Nr,

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

display(r);
display(datestr(now));

% %%%%%%%%%%%% CLUSTERING AND LABELING %%%%%%%%%%%%%%%%%%%

OUT_CL = cluster_alg(DATA,HP);

PAR_acc{r} = label_alg(DATA,OUT_CL);

STATS_acc{r} = cluster_stats_1turn(DATA,PAR_acc{r});

end

display('Finish Algorithm')
display(datestr(now));

%% RESULTS / STATISTICS

% Statistics for n turns

nSTATS = cluster_stats_nturns(STATS_acc);

%% GRAPHICS

% Quantization error (of last turn)
plot_stats_ssqe(PAR_acc{r}.SSE);

% Clusters' Prototypes and Data (of last turn)
plot_clusters_data(DATA,PAR_acc{r});

% Clusters' neigborhood and Data (of last turn)
plot_clusters_neigborhood(DATA,PAR_acc{r});

% Voronoi Cells (of last turn)
plot_clusters_voronoi(DATA,PAR_acc{r});

% Labeled Prototypes' Grid (of last turn)
plot_clusters_grid(PAR_acc{r});

%% SAVE VARIABLES AND VIDEO

% % Save All Variables
% save(OPT.file);

% % See Clusters Video (of last turn)
% if (HP.Von == 1),
%     figure;
%     movie(PAR_acc{r}.VID)
% end

% % Save Clusters Video (of last turn)
% v = VideoWriter('video.mp4','MPEG-4'); % v = VideoWriter('video.avi');
% v.FrameRate = 1;
% open(v);
% writeVideo(v,PAR_acc{r}.VID);
% close(v);

%% END