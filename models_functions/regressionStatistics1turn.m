classdef regressionStatistics1turn
    % HELP about classificationStatisticsNturns
    
    % Hyperparameters
    properties
        % ToDo - All
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        Y = [];
        Yh = [];
        error = [];
        rmse = [];
    end
    
    methods
        
        % Constructor
        function self = regressionStatistics1turn()
            % Set the hyperparameters after initializing!
        end
        
        function self = calculate_all(self,Y,Yh)
            
            self.Y = Y;
            self.Yh = Yh;
            
            [~,N] = size(Y);

            self.error = Y - Yh;
            
            self.rmse = sqrt((1/N)*sum(self.error.^2,2));
            
            
        end

    end    
    
end