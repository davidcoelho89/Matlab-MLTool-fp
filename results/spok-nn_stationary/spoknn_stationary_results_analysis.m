%% Results Analysis - Stationary Data - HPO: 0



%% Results Analysis - Stationary Data - HPO: 1

% Get number of classes from problem

[Nc,~] = size(DATA.output);

% Analysis measures

acc_median_tr = nSTATS_tr.acc_median;
acc_median_ts = nSTATS_ts.acc_median;

Analysis = zeros(Nc+10,OPT.Nr);

%accs_tr -> 1 line
%accs_ts -> 1
v1s = zeros(1,OPT.Nr);
v2s = zeros(1,OPT.Nr);
Ktypes = zeros(1,OPT.Nr);
sigmas = zeros(1,OPT.Nr);
alphas = zeros(1,OPT.Nr);
thetas = zeros(1,OPT.Nr);
gammas = zeros(1,OPT.Nr);
Nprots = zeros(Nc+1,OPT.Nr);

% Get measures

for r = 1:OPT.Nr
    
    param = PAR_acc{r};
    
    Cy = param.Cy;
    [~,Cy_seq] = max(Cy);
    for c = 1:Nc
        Nprots(c,r) = sum(Cy_seq == c);
    end
    Nprots(Nc+1,r) = length(Cy_seq);
	Ktypes = param.Ktype;
    v1s(r) = param.v1;
    v2s(r) = param.v2;
    sigmas(r) = param.sigma;
    alphas(r) = param.alpha;
    thetas(r) = param.theta;
    gammas(r) = param.gamma;
end

Nprots_median = median(Nprots,2);

% Hold analysis

Analysis(1,:) = nSTATS_tr.acc;
Analysis(2,:) = nSTATS_ts.acc;
Analysis(3,:) = v1s;
Analysis(4,:) = v2s;
Analysis(5,:) = Ktypes;
Analysis(6,:) = sigmas;
Analysis(7,:) = alphas;
Analysis(8,:) = thetas;
Analysis(9,:) = gammas;
Analysis(10:end,:) = Nprots;

% Plot Graphics

figure;
plot(Nprots(Nc+1,:),nSTATS_ts.acc,'k.');
title('Test Accuracy vs Number of Prototypes')
xlabel('Number of Prototypes')
ylabel('Test Accuracy')
axis([min(Nprots(Nc+1,:))-1 max(Nprots(Nc+1,:))+1 ...
      min(nSTATS_ts.acc)-0.01 max(nSTATS_ts.acc)+0.01])

%% END