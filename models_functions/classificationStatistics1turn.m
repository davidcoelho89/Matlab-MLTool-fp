classdef classificationStatistics1turn
    % HELP about classificationStatistics1turn
    
    % Hyperparameters
    properties
        class_type = 'bipolar';
        discretization = 0.1;
        acc = [];
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        Y = [];
        Yh = [];
        Mconf = [];
        err = [];
        roc_t = [];
        roc_tpr = [];
        roc_spec = [];
        roc_fpr = [];
        roc_prec = [];
        roc_rec = [];
        fsc = [];
        auc = [];
        mcc = [];
    end
    
    methods
        
        % Constructor
        function self = classificationStatistics1turn()
            % Set the hyperparameters after initializing!
        end
        
        function self = calculateConfusionMatrix(self,Y,Yh)
            
            [Nc,N] = size(Y);
            self.Mconf = zeros(Nc,Nc);
            
            for i = 1:N
                y_i = Y(:,i);
                yh_i = Yh(:,i);
                
                if(strcmp(self.class_type,'bipolar') || ...
                        strcmp(self.class_type,'binary'))
                    [~,iY] = max(y_i);
                    [~,iY_h] = max(yh_i);
                    self.Mconf(iY,iY_h) = self.Mconf(iY,iY_h) + 1;
                end
                
            end
            
        end
        
        function self = calculate_accuracy(self,Y,Yh)
            
            [~,N] = size(Y);
            
            if(isempty(self.Mconf))
                self.acc = 0;
                for i = 1:N
                    
                    y_i = Y(:,i);
                    [~,iY] = max(y_i);
                    yh_i = Yh(:,i);
                    [~,iY_h] = max(yh_i);
                    
                    if(iY == iY_h)
                        self.acc = self.acc + 1;
                    end
                end
                self.acc = self.acc / N;
            else
                self.acc = trace(self.Mconf)/N;
            end
            
            self.err = 1 - self.acc;
            
        end
        
        function self = calculate_error(self,Y,Yh)
            self = calculate_accuracy(self,Y,Yh);
        end
        
        function self = calculate_all(self,Y,Yh)
            
            self.Y = Y;
            self.Yh = Yh;
            self.roc_t = -1:self.discretization:1;
            
            self = self.calculateConfusionMatrix(Y,Yh);
            self = self.calculate_accuracy(Y,Yh);
            
        end
        
    end
    
end