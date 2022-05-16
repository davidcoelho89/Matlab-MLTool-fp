classdef dataNormalizer
    % HELP about dataNormalizer
    
    % Hyperparameters
    properties
        normalization = 'zscore';
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        Xmin = [];
        Xmax = [];
        Xmed = [];
        Xstd = [];
    end
    
    methods
        
        % Constructor
        function self = dataNormalizer()
            % Set the hyperparameters after initializing!
        end
        
        function self = fit(self,X)
            X = X';
            self.Xmin = min(X)';
            self.Xmax = max(X)';
            self.Xmed = mean(X)';
            self.Xstd = std(X)';
        end
        
        function Xout = transform(self,Xin)
            
            [p,N] = size(Xin);
            Xout = zeros(p,N);
            
            if(strcmp(self.normalization,'zscore'))
                for i = 1:p
                    for j = 1:N
                        Xout(i,j) = (Xin(i,j) - self.Xmed(i))/self.Xstd(i); 
                    end
                end
            else
                disp('Choose a correct option. Data was not normalized.');
                Xout = Xin;
            end
            
        end
        
        function Xout = reverse(self,Xin)

            [p,N] = size(Xin);
            Xout = zeros(p,N);
            
            if(strcmp(self.normalization,'zscore'))
                for i = 1:p
                    for j = 1:N
                        Xout(i,j) = Xin(i,j)*self.Xstd(i) + self.Xmed(i);
                    end
                end
            else
                disp('Choose a correct option. Data was not denormalized.');
                Xout = Xin;
            end
        end
        
    end
    
end
