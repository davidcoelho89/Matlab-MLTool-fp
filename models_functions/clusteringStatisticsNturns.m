classdef clusteringStatisticsNturns
    % HELP about clusteringStatisticsNturns
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
        number_of_realizations;
        realization;
        cell_of_results;
        
        ssqe_vect = [];
        msqe_vect = [];
        
        
    end
    
    methods
    
        % Constructor
        function self = clusteringStatisticsNturns(number_of_realizations)
            self.number_of_realizations = number_of_realizations;
            self.realization = 0;
            self.cell_of_results = cell(number_of_realizations,1);
        end
        
        function self = addResult(self,clusterStats1turn)
            
            self.realization = self.realization + 1;
            if(self.realization > self.number_of_realizations)
            disp('Overflow in number of results. Overwriting.')
                self.realization = 1;
            end
            
            self.cell_of_results{self.realization,1} = clusterStats1turn;

        end
        
        function self = calculate_all(self)
            
            Nr = self.number_of_realizations;
            
            self.ssqe_vect = zeros(1,Nr);
            
            for r = 1:Nr
                
                clusterStats1turn = self.cell_of_results{r,1};
                self.ssqe_vect(r) = clusterStats1turn.ssqe;
                self.msqe_vect(r) = clusterStats1turn.msqe;
                
            end
            
        end
    
    end
    
end