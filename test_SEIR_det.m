function test_SEIR_det
% Deterministic SEIR optimization

check_tt;

% Contact data
Cfixed(:,:,1) = load('SEIRData/Contact_work.txt');
Cfixed(:,:,2) = load('SEIRData/Contact_school.txt');
Cfixed(:,:,3) = load('SEIRData/Contact_other.txt');
Cfixed(:,:,4) = load('SEIRData/Contact_home.txt');

% initialize Susceptible with the total pop
S = load('SEIRData/pop.txt');

sigma = 1e-6;
nxi = 1; % don't mess this up
nmid = (nxi+1)/2;
% random variables
[beta,wbeta] = lgwt(nxi, 0.13-sigma*0.03, 0.13+sigma*0.03);   wbeta = wbeta/sum(wbeta);
[dL,wdL] = lgwt(nxi, 1.57-sigma*0.42, 1.57+sigma*0.42);       wdL = wdL/sum(wdL); 
[dC,wdC] = lgwt(nxi, 2.12-sigma*0.80, 2.12+sigma*0.80);       wdC = wdC/sum(wdC);
[dR,wdR] = lgwt(nxi, 1.54-sigma*0.40, 1.54+sigma*0.40);       wdR = wdR/sum(wdR);
[dRC,wdRC] = lgwt(nxi, 12.08-sigma*1.51, 12.08+sigma*1.51);   wdRC = wdRC/sum(wdRC);
[dD,wdD] = lgwt(nxi, 5.54-sigma*2.19, 5.54+sigma*2.19);       wdD = wdD/sum(wdD);
[rho1,wrho1] = lgwt(nxi, 0.06-sigma*0.03, 0.06+sigma*0.03);   wrho1 = wrho1/sum(wrho1);
[rho2,wrho2] = lgwt(nxi, 0.05-sigma*0.03, 0.05+sigma*0.03);   wrho2 = wrho2/sum(wrho2);
[rho3,wrho3] = lgwt(nxi, 0.08-sigma*0.04, 0.08+sigma*0.04);   wrho3 = wrho3/sum(wrho3);
[rho4,wrho4] = lgwt(nxi, 0.54-sigma*0.22, 0.54+sigma*0.22);   wrho4 = wrho4/sum(wrho4);
[rho5,wrho5] = lgwt(nxi, 0.79-sigma*0.14, 0.79+sigma*0.14);   wrho5 = wrho5/sum(wrho5);
[rhop1,wrhop1] = lgwt(nxi, 0.26-sigma*0.23, 0.26+sigma*0.23); wrhop1 = wrhop1/sum(wrhop1);
[rhop2,wrhop2] = lgwt(nxi, 0.28-sigma*0.25, 0.28+sigma*0.25); wrhop2 = wrhop2/sum(wrhop2);
[rhop3,wrhop3] = lgwt(nxi, 0.33-sigma*0.27, 0.33+sigma*0.27); wrhop3 = wrhop3/sum(wrhop3);
[rhop4,wrhop4] = lgwt(nxi, 0.26-sigma*0.11, 0.26+sigma*0.11); wrhop4 = wrhop4/sum(wrhop4);
[rhop5,wrhop5] = lgwt(nxi, 0.80-sigma*0.13, 0.80+sigma*0.13); wrhop5 = wrhop5/sum(wrhop5);
[Nin, wNin] = lgwt(nxi, 276-sigma*133, 276+sigma*133);        wNin = wNin/sum(wNin);
[alpha123, walpha123] = lgwt(nxi, 0.63-sigma*0.21, 0.63+sigma*0.21); walpha123 = walpha123/sum(walpha123);
[alpha4, walpha4] = lgwt(nxi, 0.57-sigma*0.23, 0.57+sigma*0.23); walpha4 = walpha4/sum(walpha4);
[alpha5, walpha5] = lgwt(nxi, 0.71-sigma*0.23, 0.71+sigma*0.23); walpha5 = walpha5/sum(walpha5);


% Time grid
Nt = parse_parameter('Number of time steps for control', 7);
[Tnodes,Wt] = lgwt(Nt,17,90);
[Tnodes,prm] = sort(Tnodes);
Wt = Wt(prm);

u = zeros(3,Nt);

% % Test historic control
% m(1,:) = load('SEIRData/mob_work.txt')';
% m(2,:) = load('SEIRData/mob_school.txt')';
% m(3,:) = load('SEIRData/mob_other.txt')';
% u = m(:,1:91);
% Tnodes = (0:90)';
% Wt = ones(91,1);

% u = [0.81, 0.37, 0.3, 0.37, 0.4;  % work
%      0.1*ones(1,5);  % school
%      0.83, 0.4, 0.45, 0.5, 0.55];  % other
u = 1-u;

% constraints: 0.31<= 1-u(1,:) <=1             0<= u(1,:) <= 0.69
%              0.1 <= 1-u(2,:) <=1         =>  0<= u(2,:) <= 0.9 
%              0.41<= 1-u(3,:) <=1             0<= u(3,:) <= 0.59

% D = (1/dD) \int_0^t I_{C1}(s) ds

lowerbnd = zeros(size(u));
upperbnd = [0.69*ones(1,size(u,2));
            0.90*ones(1,size(u,2));
            0.59*ones(1,size(u,2))];
        
u = min(u, upperbnd);
u = max(u, lowerbnd);
        
eps = parse_parameter('Control regularization parameter eps', 100);

cost_increase = 0;
[Cost_state,t,x] = SEIRcost(u, Nin(nmid), Tnodes, beta(nmid),dL(nmid),dC(nmid),dR(nmid),dRC(nmid),dD(nmid),[rho1(nmid);rho2(nmid);rho3(nmid);rho4(nmid);rho5(nmid)],[rhop1(nmid);rhop2(nmid);rhop3(nmid);rhop4(nmid);rhop5(nmid)],[alpha123(nmid);alpha4(nmid);alpha5(nmid)],S,Cfixed);

for iter=1:10
    Cost = Cost_state + 0.5*eps*norm((u.^2)*Wt);    
        
    Grad = SEIRFDGrad(u, Nin(nmid), Tnodes, beta(nmid),dL(nmid),dC(nmid),dR(nmid),dRC(nmid),dD(nmid),[rho1(nmid);rho2(nmid);rho3(nmid);rho4(nmid);rho5(nmid)],[rhop1(nmid);rhop2(nmid);rhop3(nmid);rhop4(nmid);rhop5(nmid)],[alpha123(nmid);alpha4(nmid);alpha5(nmid)],S,Cfixed);
    Grad = reshape(Grad, size(u));
    Grad = Grad + eps*u.*(Wt');
    Grad = u - Grad;
    Grad = max(Grad, lowerbnd);
    Grad = min(Grad, upperbnd);  
    Grad = Grad - u;
    normGrad = norm(Grad);
        
    step = 1;
    Cost_new = inf;
    while (Cost_new>Cost) && (step>1e-2)
        unew = u + step*Grad;
        [Cost_state_new,t,x] = SEIRcost(unew, Nin(nmid), Tnodes, beta(nmid),dL(nmid),dC(nmid),dR(nmid),dRC(nmid),dD(nmid),[rho1(nmid);rho2(nmid);rho3(nmid);rho4(nmid);rho5(nmid)],[rhop1(nmid);rhop2(nmid);rhop3(nmid);rhop4(nmid);rhop5(nmid)],[alpha123(nmid);alpha4(nmid);alpha5(nmid)],S,Cfixed);
        Cost_new = Cost_state_new + 0.5*eps*norm((unew.^2)*Wt);
        step = step/2;
    end
    if (Cost_new>Cost)
        cost_increase = cost_increase+1;
    end
    u = unew;
    Cost_state = Cost_state_new;
    fprintf('Iter=%d, Cost=%g, |G|=%g, step=%g\n', iter, Cost_new, normGrad, step*2);
        
    x = reshape(x, numel(t), 5, 5);
    Ic = sum(x(:,:,4)+x(:,:,5), 2);
    figure(1); plot(t,Ic); title('I_C')
    
    figure(2); plot(Tnodes, u'); title('u'); legend('work', 'school', 'other')
    
    D = sum(x(:,:,4), 2);
    D = D.*[t(2)-t(1); t(2:end)-t(1:end-1)]; D(1)=D(1)/2; D(end)=D(end)/2; D = cumsum(D);
    D = D/dD;

    if (cost_increase>5)
        break;
    end
end

% Copy vars to main space
vars = whos;
for i=1:numel(vars)
    if (exist(vars(i).name, 'var'))
        assignin('base', vars(i).name, eval(vars(i).name));
    end
end
end
