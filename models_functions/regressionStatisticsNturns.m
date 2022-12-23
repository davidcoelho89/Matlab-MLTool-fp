classdef regressionStatisticsNturns
    % HELP about regressionStatisticsNturns
    
    % Hyperparameters
    properties
        
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
        number_of_realizations;
        realization;
        cell_of_results;
        
        rmse_matrix;

    end
    
    methods
        
        % Constructor
        function self = regressionStatisticsNturns(number_of_realizations)
            self.number_of_realizations = number_of_realizations;
            self.realization = 0;
            self.cell_of_results = cell(number_of_realizations,1);
        end
        
        function self = addResult(self,sysIdStats1turn)

            self.realization = self.realization + 1;
            if(self.realization > self.number_of_realizations)
                disp('Overflow in number of results. Overwriting.')
                self.realization = 1;
            end

            self.cell_of_results{self.realization,1} = sysIdStats1turn;
            
        end
        
        function self = calculate_all(self)
            
            Nr = self.number_of_realizations;
            No = length(self.cell_of_results{1,1}.rmse);
            
            self.rmse_matrix = zeros(No,Nr);
            
            for r = 1:Nr
                stats = self.cell_of_results{r,1};
                self.rmse_matrix(:,r) = stats.rmse;
            end
            
        end
        
    end    
    
end