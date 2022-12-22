classdef classificationStatisticsNturns
    % HELP about classificationStatisticsNturns
    
	% Hyperparameters
    properties
        acc_vect = [];
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
        number_of_realizations;
        realization;
        cell_of_results;

        Mconf_sum = [];
        Mconf_mean = [];
        acc_max = [];
        acc_max_i = [];
        acc_min = [];
        acc_min_i = [];
        acc_mean = [];
        acc_median = [];
        acc_std = [];
        acc_cv = [];
        err_vect = [];
        err_max = [];
        err_max_i = [];
        err_min = [];
        err_min_i = [];
        err_mean = [];
        err_median = [];
        err_std = [];
        err_cv = [];
        fsc_vect = [];
        auc_vect = [];
        mcc_vect = [];
        
    end
    
    methods
        
        % Constructor
        function self = classificationStatisticsNturns(number_of_realizations)
            self.number_of_realizations = number_of_realizations;
            self.realization = 0;
            self.cell_of_results = cell(number_of_realizations,1);
        end

        function self = addResult(self,classStats1turn)
            
            self.realization = self.realization + 1;
            if(self.realization > self.number_of_realizations)
            disp('Overflow in number of results. Overwriting.')
                self.realization = 1;
            end
            
            self.cell_of_results{self.realization,1} = classStats1turn;

        end
        
        function self = calculate_all(self)
            
            Nr = self.number_of_realizations;
            [Nc,~] = size(self.cell_of_results{1,1}.Mconf);
            
            self.Mconf_sum = zeros(Nc,Nc);
           
            self.acc_vect = zeros(1,Nr);
            self.err_vect = zeros(1,Nr);
            
            for r = 1:Nr
                
                classStats1turn = self.cell_of_results{r,1};
                self.Mconf_sum = self.Mconf_sum + classStats1turn.Mconf;
                self.acc_vect(r) = classStats1turn.acc;
                self.err_vect(r) = classStats1turn.err;
                
            end
            
            self.Mconf_mean = self.Mconf_sum / Nr;

            [self.acc_max,self.acc_max_i] = max(self.acc_vect);
            [self.acc_min,self.acc_min_i] = min(self.acc_vect);
            self.acc_mean = mean(self.acc_vect);
            self.acc_median = median(self.acc_vect);
            self.acc_std = std(self.acc_vect);
            self.acc_cv = self.acc_std / self.acc_mean;

            [self.err_max,self.err_max_i] = max(self.err_vect);
            [self.err_min,self.err_min_i] = min(self.err_vect);
            self.err_mean = mean(self.err_vect);
            self.err_median = median(self.err_vect);
            self.err_std = std(self.err_vect);
            self.err_cv = self.err_std / self.err_mean;
            
        end
        
    end
    
    
end