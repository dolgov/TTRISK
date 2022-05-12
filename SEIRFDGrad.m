function [Grad] = SEIRFDGrad(u, Nin, Tnodes, beta,dL,dC,dR,dRC,dD,rho,rhop,alpha,S,Cfixed)
% Finite Difference gradient for the SEIR cost function
Cost = SEIRcost(u, Nin, Tnodes, beta,dL,dC,dR,dRC,dD,rho,rhop,alpha,S,Cfixed);
Grad = zeros(numel(Cost), numel(u));
for i=1:numel(u)
    du = zeros(size(u));
    du(i) = 1e-6 * max(abs(u(i)), 0.1) * (double(u(i)>=0)*2-1);
    dCost = SEIRcost(u+du, Nin, Tnodes, beta,dL,dC,dR,dRC,dD,rho,rhop,alpha,S,Cfixed);
    Grad(:,i) = (dCost-Cost)/du(i);
end
Grad = reshape(Grad, numel(Cost), size(u,1), size(u,2));
end
