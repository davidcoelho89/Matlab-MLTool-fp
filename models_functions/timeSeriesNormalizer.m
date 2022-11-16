classdef timeSeriesNormalizer
    % HELP about timeSeriesNormalizer
    %   Properties:
    %       normalization = how will be the normalization   [cte]
    %           none:    input_out = input_in
    %           binary:  normalize between [0, 1]
    %           bipolar: normalize between [-1, +1]
    %           zscore:  normalize by z-score transformation (standardzation)
    %                    (empirical mean = 0 and standard deviation = 1)
    %                    Xnorm = (X-Xmean)/(std)
    %           zscore3: Xnorm = (X-Xmean)/(3*std)
    
    % Hyperparameters
    properties
        normalization = 'zscore';
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        Umin = [];
        Umax = [];
        Umed = [];
        Ustd = [];
        Ymin = [];
        Ymax = [];
        Ymed = [];
        Ystd = [];
    end
    
    methods
        
        % Constructor
        function self = timeSeriesNormalizer()
            % Set the hyperparameters after initializing!
        end
        
        function self = fit(self,time_series)
            
            if(isfield(time_series,'input'))
                U = time_series.input';
                self.Umin = min(U)';
                self.Umax = max(U)';
                self.Umed = mean(U)';
                self.Ustd = std(U)';
            end
            
            if(isfield(time_series,'output'))
                Y = time_series.output';
                self.Ymin = min(Y)';
                self.Ymax = max(Y)';
                self.Ymed = mean(Y)';
                self.Ystd = std(Y)';
            end
            
        end
        
        function time_series_out = transform(self,time_series)
            
            % Get data matrices
            
            if(isfield(time_series,'input'))
                U = time_series.input;
                [Nu,N] = size(U);
                Unorm = zeros(Nu,N);
            end
            
            if(isfield(time_series,'output'))
                Y = time_series.output;
                [Ny,N] = size(Y);
                Ynorm = zeros(Ny,N);
            end
            
            % Normalize data matrices
            
            if(strcmp(self.normalization,'none'))
                if(isfield(time_series,'input'))
                    Unorm = U;
                end
                if(isfield(time_series,'output'))
                    Ynorm = Y;
                end
            elseif(strcmp(self.normalization,'binary'))
                if(isfield(time_series,'input'))
                    for i = 1:Nu
                        for j = 1:N
                            Unorm(i,j) = (U(i,j) - self.Umin(i))/...
                                         (self.Umax(i) - self.Umin(i));
                        end
                    end
                end
                if(isfield(time_series,'output'))
                    for i = 1:Ny
                        for j = 1:N
                            Ynorm(i,j) = (Y(i,j) - self.Ymin(i))/...
                                         (self.Ymax(i) - self.Ymin(i));
                        end
                    end
                end
            elseif(strcmp(self.normalization,'bipolar'))
                if(isfield(time_series,'input'))
                    for i = 1:Nu
                        for j = 1:N
                            Unorm(i,j) = 2*(U(i,j) - self.Umin(i))/...
                                         (self.Umax(i) - self.Umin(i)) - 1;
                        end
                    end
                end
                if(isfield(time_series,'output'))
                    for i = 1:Ny
                        for j = 1:N
                            Ynorm(i,j) = 2*(Y(i,j) - self.Ymin(i))/...
                                         (self.Ymax(i) - self.Ymin(i)) - 1;
                        end
                    end
                end
            elseif(strcmp(self.normalization,'zscore'))
                if(isfield(time_series,'input'))
                    for i = 1:Nu
                        for j = 1:N
                            Unorm(i,j) = (U(i,j) - self.Umed(i))/self.Ustd(i);
                        end
                    end
                end
                if(isfield(time_series,'output'))
                    for i = 1:Ny
                        for j = 1:N
                            Ynorm(i,j) = (Y(i,j) - self.Ymed(i))/self.Ystd(i);
                        end
                    end
                end
            elseif(strcmp(self.normalization,'zscore3'))
                if(isfield(time_series,'input'))
                    for i = 1:Nu
                        if(self.Ustd(i) == 0)
                             Unorm(i,:) = 0;
                        else
                            for j = 1:N
                                Unorm(i,j) = (U(i,j) - self.Umed(i))/(3*self.Ustd(i));
                            end                            
                        end
                    end
                end
                if(isfield(time_series,'output'))
                    for i = 1:Ny
                        if(self.Ystd(i) == 0)
                            Ynorm(i,:) = 0;
                        else
                            for j = 1:N
                                Ynorm(i,j) = (Y(i,j) - self.Ymed(i))/(3*self.Ystd(i));
                            end
                        end
                    end
                end
            else
                disp('Choose a correct option. Data was not normalized.');
                if(isfield(time_series,'input'))
                    Unorm = time_series.input;
                end
                if(isfield(time_series,'output'))
                    Ynorm = time_series.output;
                end
            end
            
            % Fill ouptut sctructure
            if(isfield(time_series,'input'))
                time_series_out.input = Unorm;
            end
            if(isfield(time_series,'output'))
                time_series_out.output = Ynorm;
            end
            
        end
        
        function time_series_out = reverse(self,time_series)
            
            if(isfield(time_series,'input'))
                Unorm = time_series.input;
                [Nu,N] = size(Unorm);
                U = zeros(Nu,N);
            end
            
            if(isfield(time_series,'output'))
                Ynorm = time_series.output;
                [Ny,N] = size(Ynorm);
                Y = zeros(Ny,N);
            end
            
            % Denormalize data matrices
            
            if(strcmp(self.normalization,'none'))
                if(isfield(time_series,'input'))
                    U = Unorm;
                end
                if(isfield(time_series,'output'))
                    Y = Ynorm;
                end
            
            elseif(strcmp(self.normalization,'binary'))
                if(isfield(time_series,'input'))
                    for i = 1:Nu
                        for j = 1:N
                            U(i,j) = Unorm(i,j)*(self.Umax(i) - self.Umin(i)) + self.Umin(i);
                        end
                    end
                end
                if(isfield(time_series,'output'))
                    for i = 1:Ny
                        for j = 1:N
                            Y(i,j) = Ynorm(i,j)*(self.Ymax(i) - self.Ymin(i)) + self.Ymin(i);
                        end
                    end
                end
                
            elseif(strcmp(self.normalization,'bipolar'))    
                if(isfield(time_series,'input'))
                    for i = 1:Nu
                        for j = 1:N
                            U(i,j) = 0.5*(Unorm(i,j) + 1)*(self.Umax(i) - self.Umin(i)) + self.Umin(i);
                        end
                    end
                end
                if(isfield(time_series,'output'))
                    for i = 1:Ny
                        for j = 1:N
                            Y(i,j) = 0.5*(Ynorm(i,j) + 1)*(self.Ymax(i) - self.Ymin(i)) + self.Ymin(i);
                        end
                    end
                end
                
            elseif(strcmp(self.normalization,'zscore'))
                if(isfield(time_series,'input'))
                    for i = 1:Nu
                        for j = 1:N
                            U(i,j) = Unorm(i,j)*self.Ustd(i) + self.Umed(i);
                        end
                    end
                end
                if(isfield(time_series,'output'))
                    for i = 1:Ny
                        for j = 1:N
                            Y(i,j) = Ynorm(i,j)*self.Ystd(i) + self.Ymed(i);
                        end
                    end
                end
                
            elseif(strcmp(self.normalization,'zscore3'))
                if(isfield(time_series,'input'))
                    for i = 1:Nu
                        for j = 1:N
                            U(i,j) = Unorm(i,j)*self.Ustd(i)*3 + self.Umed(i);
                        end
                    end
                end
                if(isfield(time_series,'output'))
                    for i = 1:Ny
                        for j = 1:N
                            Y(i,j) = Ynorm(i,j)*self.Ystd(i)*3 + self.Ymed(i);
                        end
                    end
                end
                
            else
                disp('Choose a correct option. Data was not normalized.')
                if(isfield(time_series,'input'))
                    U = Unorm;
                end
                if(isfield(time_series,'output'))
                    Y = Ynorm;
                end

            end
            
            % Fill ouptut sctructure
            if(isfield(time_series,'input'))
                time_series_out.input = U;
            end
            if(isfield(time_series,'output'))
                time_series_out.output = Y;
            end
            
        end
        
    end
    
end