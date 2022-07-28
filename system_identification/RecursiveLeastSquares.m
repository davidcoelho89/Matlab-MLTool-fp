classdef RecursiveLeastSquares 
    properties
        forgetting_rate
        inv_corr_matrix
        weight
        residue
        error
        k_gain
    end

    methods
        function self = Init(self, num_in, num_out, forgetting_rate, reg_const)
            self.forgetting_rate = forgetting_rate;
            self.inv_corr_matrix = reg_const * eye(num_in);
            self.weight = zeros(num_in, num_out);
        end
        function self = Update(self, in, out, target)
            self.error(:, end+1) = target - out;
            self.k_gain = (self.inv_corr_matrix * in) / (self.forgetting_rate + (in' * self.inv_corr_matrix * in));
            self.weight = self.weight + self.k_gain * self.error(:, end)';
            %size(self.weight)
            self.inv_corr_matrix = (1 / self.forgetting_rate) * ...
                (self.inv_corr_matrix - self.k_gain * in' * self.inv_corr_matrix);
            self.residue(:, end+1) = target - self.weight' * in;
        end
    end
end
