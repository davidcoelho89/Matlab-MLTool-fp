NAMES = {'Linear','Gaussian',... 
         'Polynomial', 'Exponential',...
         'Cauchy', 'Log',...
         'Sigmoid', 'Kmod'};

nstats_all_tr = variables.nstats_all_tr;

class_stats_ncomp(nstats_all_tr,NAMES);

nstats_all_ts = variables.nstats_all_ts;

class_stats_ncomp(nstats_all_ts,NAMES);
