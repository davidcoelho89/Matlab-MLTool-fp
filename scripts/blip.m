classdef blip
    % Summary of the class
    % Hehe
    
    properties
        AoA(1,1) double {mustBeInteger, mustBePositive} = 2;
    end
    
    properties (GetAccess = public, SetAccess = protected)
        signal(1,1) double {mustBeInteger, mustBePositive} = 3;
    end
    
    properties (Constant, Hidden)
        range = 1;
    end

    properties (Dependent)
        wavelength;
    end
    
    methods
        
        % Can have getter functions!
        function wavelength = get.wavelength(self)
            wavelength = self.AoA / self.signal;
        end
        
        function self = blip(AoA,signal)
            if nargin == 2
                self.AoA = AoA;
                self.signal = signal;
            end
        end
        
        function identify(self)
            disp("A " + self.AoA + " degree");
        end
        
    end
    
end
