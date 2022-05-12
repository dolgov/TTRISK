% Function to compute nonzeros of A on given random val xi
function [a] = Anonzeros(model,i,j,xi)
aKL=model.random.bases*sqrt(model.random.sv)*xi; % K-L expanssion

% Lognormal transformation
model.random.lambdaf = log((model.random.a_mean.^2)./sqrt(model.random.a_sigma.^2+model.random.a_mean.^2));
model.random.epsilonf = sqrt(log(model.random.a_sigma.^2./(model.random.a_mean.^2)+1));
aKL_log=exp(model.random.lambdaf+model.random.epsilonf.*aKL);

% a = aKL_log';

[model_adaptive]=Assemble_A(model,aKL_log);       % Impose random field on A.
[model_adaptive]=Dirichlet_BC(model_adaptive);    % Impose Dirichlet BC.
a = model_adaptive.Ab(i+(j-1)*size(model_adaptive.Ab,1));
a = full(a)';
end
