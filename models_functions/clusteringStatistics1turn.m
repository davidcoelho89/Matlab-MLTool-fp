classdef clusteringStatistics1turn
    % HELP about clusteringStatistics1turn
    
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        ssqe = [];
        msqe = [];
        aic = [];
        bic = [];
        ch = [];
        db = [];
        dunn = [];
        fpe = [];
        mdl = [];
        silhouette = [];
    end

    methods
        
        % Constructor
        function self = clusteringStatistics1turn()
            % Set the hyperparameters after initializing!
        end
        
        function self = calculate_ssqe(self,model,X)
            self.ssqe = 0;
            self.msqe = 0;
            
            [~,N] = size(X); 
            for n = 1:N
                winner = model.Yh(n);
                self.ssqe = self.ssqe + sum((X(:,n) - model.Cx(:,winner)).^2);
            end
            self.msqe = self.msqe + self.ssqe / N;
        end
        
        function self = calculate_all(self,model,X)
            
            self = self.calculate_ssqe(model,X);
            
            % ToDo - other measures
            %aic
            %bic
            %ch
            %db
            %dunn
            %fpe
            %mdl
            %sil

        end
        
    end
















end