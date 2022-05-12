function [y]=hess_logsmooth(x, eps)
% Second derivative of the logsigmoid function
z = x/eps;
y = (1/eps)./(exp(-z/2) + exp(z/2)).^2;
end
