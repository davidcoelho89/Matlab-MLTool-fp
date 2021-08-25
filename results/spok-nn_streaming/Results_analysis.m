%% Results Analysis - Streaming Data - HPO: 0

% clear;
% clc;

% Analysis measures

Analysis = zeros(8,OPT.Nr);

Accs = zeros(1,OPT.Nr);
Nprots = zeros(1,OPT.Nr);
v1s = zeros(1,OPT.Nr);
Ktypes = zeros(1,OPT.Nr);
sigmas = zeros(1,OPT.Nr);
alphas = zeros(1,OPT.Nr);
thetas = zeros(1,OPT.Nr);
gammas = zeros(1,OPT.Nr);

% Get measures

for r = 1:OPT.Nr
    acc_vect = accuracy_vector_acc{r};
    Accs(r) = acc_vect(end);
    
    param = PAR_acc{r};
    [~,Nprots(r)] = size(param.Cx);
    v1s(r) = param.v1;
    Ktypes = param.Ktype;
    sigmas(r) = param.sigma;
    alphas(r) = param.alpha;
    thetas(r) = param.theta;
    gammas(r) = param.gamma;
end

% Hold analysis

Analysis(1,:) = Accs;
Analysis(2,:) = Nprots;
Analysis(3,:) = v1s;
Analysis(4,:) = Ktypes;
Analysis(5,:) = sigmas;
Analysis(6,:) = alphas;
Analysis(7,:) = thetas;
Analysis(8,:) = gammas;

% Plot Graphics

plot(Nprots,Accs,'k.');

% Verify best accuracies

% indexes = find(Accs == max(Accs)),
% v1s(indexes),
% Nprots(indexes),
% Ktypes(indexes),
% sigmas(indexes),
% alphas(indexes),
% thetas(indexes),
% gammas(indexes),

%% Results Analysis - Streaming Data - HPO: 1

x = 1:Nttt;

% Data and Prototypes
figure;
hold on 
plot(DATAttt.input(1,:),DATAttt.input(2,:),'r.');
plot(PAR.Cx(1,:),PAR.Cx(2,:),'k*');
title('Data and Prototypes')
hold off

% Number of samples per class
figure;
colors = lines(Nc);
hold on
for c = 1:Nc
    plot(x,samples_per_class(c,:),'Color',colors(c,:));
end
title('Number of Samples Per Class')
hold off

% Number of Prototypes (Total and per class)
figure;
colors = lines(Nc+1);
hold on
for c = 1:Nc+1
    plot(x,prot_per_class(c,:),'Color',colors(c,:));
end
title('Number of Prototypes')
hold off

% Number of hits x number of errors
figure;
hold on
plot(x,no_of_errors,'r-');
plot(x,no_of_correct,'b-');
title('number of hits and errors')
hold off

% Percentage of Correct Classified
figure;
hold on
plot(x,accuracy_vector,'r-');
title('Percentage of correct classified')
xlabel('Time Step')
ylabel('Accuracy')
axis([-1 length(x) -0.1 1.1])
hold off

% Percentage of Misclassified
figure;
hold on
plot(x,1-accuracy_vector,'r-');
title('Percentage of samples misclassified')
xlabel('Time Step')
ylabel('Error Rate')
axis([-1 length(x) -0.1 1.1])
hold off

% Ktype = PAR.Ktype;
Acc = accuracy_vector(end);
Err = 1 - Acc;
[~,Nprot] = size(PAR.Cx);
% v1 = PAR.v1;
% sigma = PAR.sigma;
% alpha = PAR.alpha;
% theta = PAR.theta;
% gamma = PAR.gamma;

% vetor = [Ktype,Acc,Err,Nprot,v1,sigma,alpha,theta,gamma];
% display(vetor);

% OUTttt.y_h = predict_vector;
% STATS = class_stats_1turn(DATAttt,OUTttt);

%% Results Analysis - Stationary Data - HPO: 1

[~,A] = max(DATA.output);
edges = unique(A);
counts1 = histc(A(:), edges);

[~,A] = max(DATAhpo.output);
edges = unique(A);
counts2 = histc(A(:), edges);

DATA = DATAhpo;
HPgs = HP_gs;
f_train = @isk2nn_train;
f_class = @isk2nn_classify;
PSp = GSp;
nargin = 5;

[~,test] = max(PAR.Cy);
he = sum(test == 1);

%% END