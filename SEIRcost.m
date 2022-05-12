function [Cost,t,x] = SEIRcost(u, Nin, Tnodes, beta,dL,dC,dR,dRC,dD,rho,rhop,alpha,S,Cfixed)
% Cost function of the SEIR model

% initialize infected
Nin = Nin .* [0.1; 0.4; 0.35; 0.1; 0.05];

% Severity of infection
Ein = Nin/3;
I1in = rho.*Nin.*2/3;
I2in = (1-rho).*Nin.*2/3;
IC1in = zeros(5,size(Nin,2));
IC2in = zeros(5,size(Nin,2));

% Initial state
x0 = [Ein; I1in; I2in; IC1in; IC2in];
x0 = x0(:);

dt = 0.1;

t = (0:dt:90)';
x = SEIRIEuler(dt,90/dt,x0, u, Tnodes, beta,dL,dC,dR,dRC,dD,rho,rhop,alpha,S,Cfixed);

x = reshape(x, numel(t), 5, 5, []);
D = sum(x(:,:,4,:), 2);
D = D.*[t(2)-t(1); t(2:end)-t(1:end-1)]; D(1,:,:,:)=D(1,:,:,:)/2; D(end,:,:,:)=D(end,:,:,:)/2; D = sum(D);
D = D./reshape(dD, 1,1,1,[]);

Phi = max(sum(x(:,:,4,:) + x(:,:,5,:), 2) - 1e4, 0);
Phi = Phi.*[t(2)-t(1); t(2:end)-t(1:end-1)]; Phi(1,:,:,:)=Phi(1,:,:,:)/2; Phi(end,:,:,:)=Phi(end,:,:,:)/2; Phi = sum(Phi);

% u = reshape(u, 3, []);
Cost = (D + Phi)/2;
Cost = Cost(:);
end
