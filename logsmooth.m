function [y]=logsmooth(x, eps)
% Logsigmoid smoothing function
y = eps*log(1 + exp(x/eps));
y(isnan(y)|isinf(y)) = x(isnan(y)|isinf(y));
end
