classdef regressionStatisticsNturns
    % HELP about sysIdStatisticsNturns
    
    % Hyperparameters
    properties
        
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
        number_of_repetitions = 0;
        cell_of_results = [];
        
        rmse_vect = [];

    end
    
    methods
        
        % Constructor
        function self = regressionStatisticsNturns()
            % Set the hyperparameters after initializing!
        end
        
        function self = addResult(self,sysIdStats1turn)
            
            Nr = self.number_of_repetitions + 1;
            self.number_of_repetitions = Nr;
            
            if(self.number_of_repetitions == 1)
                self.cell_of_results = cell(Nr,1);
            else
                cell_aux = self.cell_of_results;
                self.cell_of_results = cell(Nr,1);
                self.cell_of_results(1:Nr-1,1) = cell_aux;
            end
            
            self.cell_of_results{Nr,1} = sysIdStats1turn;
            
        end
        
        function self = calculate_all(self)
            
            Nr = self.number_of_repetitions;
            No = length(self.cell_of_results{1,1}.rmse);
            
            self.rmse_vect = zeros(No,Nr);
            
            for r = 1:Nr
                stats = self.cell_of_results{r,1};
                self.rmse_vect(:,r) = stats.rmse;
            end
            
        end
        
    end    
    
end