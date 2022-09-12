classdef regressionStatisticsNturns
    % HELP about sysIdStatisticsNturns
    
    % Hyperparameters
    properties
        
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
        number_of_realizations;
        repetition;
        cell_of_results;
        
        rmse_matrix;

    end
    
    methods
        
        % Constructor
        function self = regressionStatisticsNturns(number_of_realizations)
            self.number_of_realizations = number_of_realizations;
            self.repetition = 0;
            self.cell_of_results = cell(number_of_realizations,1);

        end
        
        function self = addResult(self,sysIdStats1turn)

            self.repetition = self.repetition + 1;
            self.cell_of_results{self.repetition,1} = sysIdStats1turn;
            
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