function [x] = SEIRIEuler(dt,Nt,x0, u, Tnodes, beta,dL,dC,dR,dRC,dD,rho,rhop,alpha,S,Cfixed)
% Simple Implicit Euler method for SEIR model
x = repmat(reshape(x0, 1,25,[]), Nt+1, 1, 1);
% C_{i,j} = \sum_{k=1}^{3} \alpha_i * (1-m_k(t)) * C_{i,j}^k + 1 C_{i,j}^h
% we use the model m_k(t) = L_j(t) u_{k,j}, where L_j are Legendres
u = reshape(u, 3, []);

beta = reshape(beta, 1,1,[]);
kappa = reshape(1./dL, 1,1,[]);
gammaC = reshape(1./dC, 1,1,[]);
gammaR = reshape(1./dR, 1,1,[]);
gammaRC = reshape(1./dRC, 1,1,[]);
nu = reshape(1./dD, 1,1,[]);
rho = reshape(rho, 5, 1, []);
rhop = reshape(rhop, 5, 1, []);
alpha = reshape(alpha, 3, 1, []);

SN = reshape(S,5,1)./reshape(S,1,5);
I = speye(5);

% time-invariant part
J0 = sparse(25*numel(beta), 25*numel(beta));
for i=1:numel(beta)
    J0(1+(i-1)*25:i*25, 1+(i-1)*25:i*25) = ...
        [-kappa(1,1,i).*I,         sparse(5,5),   sparse(5,5), sparse(5,5), sparse(5,5);
        rho(:,:,i).*kappa(1,1,i).*I,     -gammaC(1,1,i).*I,      sparse(5,5),      sparse(5,5), sparse(5,5);
        (1-rho(:,:,i)).*kappa(1,1,i).*I, sparse(5,5),        -gammaR(1,1,i).*I,    sparse(5,5), sparse(5,5);
        sparse(5,5),          rhop(:,:,i).*gammaC(1,1,i).*I, sparse(5,5),      -nu(1,1,i).*I,   sparse(5,5);
        sparse(5,5),      (1-rhop(:,:,i)).*gammaC(1,1,i).*I, sparse(5,5),      sparse(5,5), -gammaRC(1,1,i).*I;
        ];
end


for it=1:Nt
    t = it*dt;
    if (t<17)
        C = repmat(Cfixed(:,:,4), 1,1,size(x,3));
        C(1:3,:,:) = C(1:3,:,:) + sum(Cfixed(1:3,:,1:3), 3);
        C(4,:,:) = C(4,:,:) + sum(Cfixed(4,:,1:3), 3);
        C(5,:,:) = C(5,:,:) + sum(Cfixed(5,:,1:3), 3);
    else
        m = u*lagrange_interpolant(Tnodes, t)';
        m = 1-m;
        m = reshape(m, 1, 1, 3);
        C = repmat(Cfixed(:,:,4), 1,1,size(x,3));
        C(1:3,:,:) = C(1:3,:,:) + alpha(1,:,:).*sum(Cfixed(1:3,:,1:3).*m, 3);
        C(4,:,:) = C(4,:,:) + alpha(2,:,:).*sum(Cfixed(4,:,1:3), 3);
        C(5,:,:) = C(5,:,:) + alpha(3,:,:).*sum(Cfixed(5,:,1:3), 3);
    end
    
    indi =  (1:5)' + zeros(1,10) + 25*reshape(0:numel(beta)-1, 1,1,[]);  indi = indi(:);
    indj = zeros(5,1) + (6:15) + 25*reshape(0:numel(beta)-1, 1,1,[]);  indj = indj(:);
    Jc = beta.*SN.*C;
    Jc = repmat(Jc, 1,2,1);
    Jc = sparse(indi, indj, Jc(:), 25*numel(beta), 25*numel(beta));
    J = Jc+J0;
    A = speye(25*numel(beta)) - dt*J;
    f = reshape(x(it,:,:), [], 1);
    x(it+1,:,:) = reshape(A\f, 25, numel(beta));
end
end
