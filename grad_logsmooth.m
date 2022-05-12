function [y]=grad_logsmooth(x, eps)
% Derivative of the logsigmoid function
z = x/eps;
y = 1./(exp(-z)+1);
end
