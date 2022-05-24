classdef classificationStatisticsNturns
    % HELP about classificationStatisticsNturns
    
	% Hyperparameters
    properties
        acc_vect = [];
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
        number_of_repetitions = 0;
        cell_of_results = [];

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
        
        function self = addResult(self,classStats1turn)
            
            Nr = self.number_of_repetitions + 1;
            self.number_of_repetitions = Nr;

            disp('add result!');
            disp('number of repetitions: ');
            disp(self.number_of_repetitions);
            
            if(self.number_of_repetitions == 1)
                self.cell_of_results = cell(Nr,1);
            else
                cell_aux = self.cell_of_results;
                self.cell_of_results = cell(Nr,1);
                self.cell_of_results(1:Nr-1,1) = cell_aux;
            end
            self.cell_of_results{Nr,1} = classStats1turn;
        end
        
        function self = calculate(self)
            
            Nr = self.number_of_repetitions;
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
            
        end
        
    end
    
    
end











