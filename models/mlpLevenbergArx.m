classdef mlpLevenbergArx
    
    % Hyperparameters
    properties
        
        neuron   = 5;
        minMSE = 1; 
        minGRAD = 1e-1;
        epoch = 1000;
        output_order = 2;
        input_order = 2;
        LowerLimit = -100;
        UpperLimit =  100;
        umin = 1e-10;
        umax = 1e+10;
        uscale = 10;
        
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
        Win = [];
        Wout = [];
        bin = [];
        bout = [];
        nin = [];
        Yh = [];
        
    end
    
    methods
        
        % Constructor
        function self = mlpLevenbergArx()
            % Set the hyperparameters after initializing!
        end
        
        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            [feature, ~] = size(X); % sample is # of data, feature is # of input
            [nout,~] = size(Y); % outcol is # of output
            
            self.nin = self.neuron;
            self.Win  = rand(self.nin, feature); % input layer weight matrix
            self.bin  = rand(self.nin, 1); % input layer bias matrix
            self.Wout = rand(nout, self.nin); %output layer weight matrix
            self.bout = rand(nout, 1);  %output layer bias matrix.
            
            [~, error,netout,netin] = ...
                    self.yprediction(self,X,self.Win,self.bin,self.Wout,self.bout,Y);
            
            costx = 0.5*sum(error.^2);

            iter = 0;
            uK = 1.01;
            
            loop1 = 1;
            while loop1 && iter < self.epoch
                
                iter = iter +1;
                
                param = self.vectorizationGradient(self.Win,self.Wout,...
                                                   self.bin,self.bout,...
                                                   self.nin,feature,nout);
                J = self.findJacobian(self,netout,netin,self.Wout,X,self.nin,feature,nout);
                
                loop2 = 1; 
                while loop2
                    pk = -pinv(J'*J + uK*eye(size(J,2),size(J,2)))*J'*error';
                    zk = param + pk;
                    [self.Win,self.Wout,self.bin,self.bout] = ...
                        self.devectorization(zk,self.nin,nout,feature);
                    [ ~, error,netout,netin] = ...
                        self.yprediction(self,X,self.Win,self.bin,self.Wout,self.bout,Y);
                    costz = 0.5*sum(error.^2);
                    
                    if costz < costx
                        
                        [~,sk,~,~,~,~] = self.goldenSection(self,self.LowerLimit,self.UpperLimit,1e-10,Y,self.nin,param,pk,feature,X);
                        % sk = newtonRhapson(param,pk,X,Y,nin,feature);
                        param = param + sk*pk;
                        
                        [self.Win,self.Wout,self.bin,self.bout] = ...
                            self.devectorization(zk,self.nin,nout,feature);
                        [~, error,netout,netin] = ...
                            self.yprediction(self,X,self.Win,self.bin,self.Wout,self.bout,Y);
                        costx = sum(error.^2);
                        uK = uK/self.uscale;
                        loop2 = 0;
                    else
                        uK = uK*self.uscale;
                    end
                    
                    if (uK < self.umin) || (uK > self.umax)
                        loop1 = 0;
                        loop2 = 0;
                        disp('Hessian matrix is singular');
                    end
                    
                end
                
                costx = 0.5*sum(error.^2);
                
                clc;
                disp(costx);
                
                if costx < self.minMSE && norm(2*J'*error') < self.minGRAD
                    loop1 = 0;
                    disp('Minimum training error (MSE) and Gradient vector norm are satisfied');
                end
                
            end
            
            if iter >= self.epoch
                disp('Max. iteration condition is satisfied');
            end
            
            [pred,~,~,~] = self.yprediction(self,X,self.Win,self.bin,self.Wout,self.bout,Y);
            
        end
        
        function self = predict(self,X)
            netin = self.Win*X + self.bin;
            netout = self.Wout*self.h(netin) + self.bout;
            self.Yh = self.hOut(netout);
        end
        
    end
    
    methods (Static)
        
        function y = h(x)
            y = tansig(x);
        end
        
        function y = hOut(x)
            y = purelin(x);
        end
        
        function y = hprime(x)
            y = (1-tansig(x).^2);
        end
        
        function y = hprimeOut(x)
            y = ones(size(x));
        end
        
        function [ypred,error,netout,netin] = yprediction(self,X,Win,bin,Wout,bout,Y)
            netin = Win*X + bin;
            netout = Wout*self.h(netin) + bout;
            ypred = self.hOut(netout);
            error = Y - ypred;
        end
        
        function [ypred,netout,netin] = yprediction2(self,X,Win,bin,Wout,bout)
            netin = Win*X + bin;
            netout = Wout*self.h(netin) + bout;
            ypred = self.hOut(netout);
        end
        
        function [y] = vectorizationGradient(gradWin,gradWout,gradbin,gradbout,N,feature,numout)
            y = [];
            y = [y; reshape(gradWin, [N*feature,1])];
            y = [y; reshape(gradbin, [N, 1])];
            y = [y; reshape(gradWout,[N*numout, 1])];
            y = [y; reshape(gradbout,[numout, 1])];
        end
        
        function [Win,Wout,bin,bout] = devectorization(vector,N,numout,feature)

            Win = reshape(vector(1:feature*N), [N, feature]);
            vector = vector(feature*N+1:end);

            bin = reshape(vector(1:N), [N, 1]);
            vector = vector(N+1:end);

            Wout = reshape(vector(1:numout*N), [numout, N]);
            vector = vector((numout*N)+1:end);

            bout = reshape(vector(1:end), [numout, 1]);

        end
        
        function J = findJacobian(self,netout, netin, Wout, X, nin, feature, nout)

            J = zeros(size(X,2), feature*nin + nin + nout*nin + nout);
            dWout = zeros(nout, nin);
            dWin  = zeros(nin, feature);
            dbout = zeros(nout, 1);
            dbin  = zeros(nin, 1);
            for i = 1 : size(X,2)
                dWout = self.findGradWout(self,netout(:,i),netin(:,i));
                dbout = sum(self.findGradbout(self,netout(:,i)),2);
                dWin = self.findGradWin(self,netout(:,i),Wout,netin(:,i),X(:,i));
                dbin = sum(self.findGradbin(self,netout(:,i),netin(:,i),Wout),2);
                grad = self.vectorizationGradient(dWin,dWout,dbin,dbout,nin,feature,nout);
                J(i,:) = grad;
            end

        end
        
        function y = findGradWout(self,netout,netin)
            y = -(self.hprimeOut(netout))*self.h(netin)';
        %     y = -self.h(netin)';
        end
        
        function y = findGradbout(self,netout)
        %     y = -self.hprimeOut(netout);
            y = -1;
        end
        
        function y = findGradWin(self,netout,Wout,netin,X)
        %     y = -((Wout'*(self.hprimeOut(netout)).*self.hprime(netin)))*X';
            y = -((Wout'.*self.hprime(netin)))*X';
        end
        
        function y = findGradbin(self,netout,netin,Wout)
        %     y = -((Wout'*(self.hprimeOut(netout)).*self.hprime(netin)));
            y = -Wout'.*self.hprime(netin);
        end
        
        function [t1,t2,ft1,ft2,N,tolerance] = ...
          goldenSection(self,tLowerLimit,tUpperLimit,tFinalPoint,y,s,tk,pk,R,t)
  
            tao = 0.38197;    
            tolerance = tFinalPoint/(tUpperLimit - tLowerLimit);    
            N = floor(-2.078*log(tolerance));
            
            t1 = tLowerLimit + tao*(tUpperLimit - tLowerLimit);
            [Win,Wout,bin,bout] = self.devectorization(tk+t1*pk,s,size(y,1),R);
            [prediction,~,~] = self.yprediction2(self,t,Win,bin,Wout,bout);
            ft1 = 0.5*(((y-prediction).^2));

            t2 = tUpperLimit - tao*(tUpperLimit - tLowerLimit);
            [Win,Wout,bin,bout] = self.devectorization(tk+t2*pk,s,size(y,1),R);
            [prediction,~,~] = self.yprediction2(self,t,Win,bin,Wout,bout);
            ft2 = 0.5*(((y-prediction).^2));

            ft1 = sum(ft1);
            ft2 = sum(ft2);

            k = 0;    

            for i = 1:N
             if k < N
                if ft1 > ft2
                    tLowerLimit = t1;
                    t1 = t2;
                    ft1 = ft2;
                    t2 = tUpperLimit - tao*(tUpperLimit - tLowerLimit);
                    [Win,Wout,bin,bout] = self.devectorization(tk+t2*pk,s,size(y,1),R);
                    [prediction,~,~] = self.yprediction2(self,t,Win,bin,Wout,bout);
                    ft2 = 0.5*(((y-prediction).^2));
                    ft2 = sum(ft2);
                    k = k + 1;
                elseif ft1 < ft2            
                    tUpperLimit = t2;
                    t2 = t1;           
                    ft2 = ft1;           
                    t1 = tLowerLimit + tao*(tUpperLimit - tLowerLimit);
                    [Win,Wout,bin,bout] = self.devectorization(tk+t1*pk,s,size(y,1),R);
                    [prediction,~,~] = self.yprediction2(self,t,Win,bin,Wout,bout);
                    ft1 = 0.5*(((y-prediction).^2));
                    ft1 = sum(ft1);
                    k = k + 1;          
                end        
             else       
                break;
             end  
            end
            
        end % end golden section
        
    end % end methods (Static)
    
end % end classdef