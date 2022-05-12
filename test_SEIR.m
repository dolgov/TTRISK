function test_SEIR
% SEIR example with CVaR with respect to random parameters

check_tt;

% Contact data
Cfixed(:,:,1) = load('SEIRData/Contact_work.txt');
Cfixed(:,:,2) = load('SEIRData/Contact_school.txt');
Cfixed(:,:,3) = load('SEIRData/Contact_other.txt');
Cfixed(:,:,4) = load('SEIRData/Contact_home.txt');

% initialize Susceptible with the total pop
S = load('SEIRData/pop.txt');

sigma = parse_parameter('Standard deviation of random variables sigma', 1e-1);
nxi = parse_parameter('Number of grid points in random variables', 3);
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
beta_cvar = parse_parameter('CVaR quantile beta', 0.5);
eps_final = parse_parameter('Smoothing parameter', 1000);

tol = parse_parameter('TT approximation tolerance', 1e-2);

for irun=1:100
    % Sample random var for a simple uncontrolled stats
    sbeta = 0.13+sigma*0.03*(rand*2-1);
    sdL = 1.57+sigma*0.42*(rand*2-1);
    sdC = 2.12+sigma*0.80*(rand*2-1);
    sdR = 1.54+sigma*0.40*(rand*2-1);
    sdRC = 12.08+sigma*1.51*(rand*2-1);
    sdD = 5.54+sigma*2.19*(rand*2-1);
    srho1 = 0.06+sigma*0.03*(rand*2-1);
    srho2 = 0.05+sigma*0.03*(rand*2-1);
    srho3 = 0.08+sigma*0.04*(rand*2-1);
    srho4 = 0.54+sigma*0.22*(rand*2-1);
    srho5 = 0.79+sigma*0.14*(rand*2-1);
    srhop1 = 0.26+sigma*0.23*(rand*2-1);
    srhop2 = 0.28+sigma*0.25*(rand*2-1);
    srhop3 = 0.33+sigma*0.27*(rand*2-1);
    srhop4 = 0.26+sigma*0.11*(rand*2-1);
    srhop5 = 0.80+sigma*0.13*(rand*2-1);
    sNin = 276+sigma*133*(rand*2-1);
    salpha123 = 0.63+sigma*0.21*(rand*2-1);
    salpha4 = 0.57+sigma*0.23*(rand*2-1);
    salpha5 = 0.71+sigma*0.23*(rand*2-1);
    [Cost_state,t,xhist(:,:,:,irun)] = SEIRcost(u, sNin, Tnodes, sbeta,sdL,sdC,sdR,sdRC,sdD,[srho1;srho2;srho3;srho4;srho5],[srhop1;srhop2;srhop3;srhop4;srhop5],[salpha123;salpha4;salpha5],S,Cfixed);
end
Ichist = sum(xhist(:,:,4,:)+xhist(:,:,5,:), 2);
Icmean = mean(Ichist, 4);
Icstd = std(Ichist, [], 4);

Xi = tt_meshgrid_vert(tt_tensor(Nin), tt_tensor(beta),tt_tensor(dL),tt_tensor(dC),tt_tensor(dR),tt_tensor(dRC),tt_tensor(dD),tt_tensor(rho1),tt_tensor(rho2),tt_tensor(rho3),tt_tensor(rho4),tt_tensor(rho5),tt_tensor(rhop1),tt_tensor(rhop2),tt_tensor(rhop3),tt_tensor(rhop4),tt_tensor(rhop5),tt_tensor(alpha123),tt_tensor(alpha4),tt_tensor(alpha5));
W = mtkron({tt_tensor(wNin), tt_tensor(wbeta),tt_tensor(wdL),tt_tensor(wdC),tt_tensor(wdR),tt_tensor(wdRC),tt_tensor(wdD),tt_tensor(wrho1),tt_tensor(wrho2),tt_tensor(wrho3),tt_tensor(wrho4),tt_tensor(wrho5),tt_tensor(wrhop1),tt_tensor(wrhop2),tt_tensor(wrhop3),tt_tensor(wrhop4),tt_tensor(wrhop5),tt_tensor(walpha123),tt_tensor(walpha4),tt_tensor(walpha5)});

Cost_state = 4; Vg = 4; Grad = 4; V = 4; Vh = 4;
tic;
Cost_state = amen_cross_s(Xi, @(x)SEIRcost(u, x(:,1)', Tnodes, x(:,2)',x(:,3)',x(:,4)',x(:,5)',x(:,6)',x(:,7)',[x(:,8)';x(:,9)';x(:,10)';x(:,11)';x(:,12)'],[x(:,13)';x(:,14)';x(:,15)';x(:,16)';x(:,17)'],[x(:,18)';x(:,19)';x(:,20)'],S,Cfixed), tol, 'y0', Cost_state);
toc;

cost_increase = 0;

t_cvar = dot(W,Cost_state)
eps_cvar = t_cvar;
t_cvar = 0;

for iter=1:20
    % Derivatives of the smoother at the given g
    Vg = amen_cross_s({Cost_state}, @(x)grad_logsmooth(x-t_cvar,eps_cvar), tol, 'y0', Vg);
    V = amen_cross_s({Cost_state}, @(x)logsmooth(x-t_cvar,eps_cvar), tol, 'y0', V);
    cv = t_cvar + dot(W,V) / (1-beta_cvar);    
    Cost = cv + 0.5*eps*norm((u.^2)*Wt);
    
    Vh = amen_cross_s({Cost_state}, @(x)hess_logsmooth(x-t_cvar,eps_cvar), tol, 'y0', Vh);
    Vhmean = dot(W,Vh)/(1-beta_cvar);
    
    Grad = amen_cross_s(Xi, @(x)reshape(SEIRFDGrad(u, x(:,1)', Tnodes, x(:,2)',x(:,3)',x(:,4)',x(:,5)',x(:,6)',x(:,7)',[x(:,8)';x(:,9)';x(:,10)';x(:,11)';x(:,12)'],[x(:,13)';x(:,14)';x(:,15)';x(:,16)';x(:,17)'],[x(:,18)';x(:,19)';x(:,20)'],S,Cfixed), [], numel(u)), tol, 'kickrank', 0, 'y0', Grad, 'exitdir', 1, 'dir', -1, 'nswp', 2);
    
    % CVaR computation
    grad_J_u = reshape(dot(Grad, W.*Vg) / (1-beta_cvar), size(u)) + eps*u.*(Wt');
    grad_J_t = 1 - dot(W,Vg) / (1-beta_cvar);
    
    grad_J_u = u - grad_J_u;
    grad_J_u = max(grad_J_u, lowerbnd);
    grad_J_u = min(grad_J_u, upperbnd);  
    grad_J_u = grad_J_u - u;
    normGrad = norm(grad_J_u);
    
    
    % Line search
    step = 1;
    Cost_new = inf;
    while (Cost_new>Cost) && (step>1e-2)
        unew = u + step*grad_J_u;
        tnew = t_cvar - step*grad_J_t / Vhmean;
        Cost_state_new = amen_cross_s(Xi, @(x)SEIRcost(unew, x(:,1)', Tnodes, x(:,2)',x(:,3)',x(:,4)',x(:,5)',x(:,6)',x(:,7)',[x(:,8)';x(:,9)';x(:,10)';x(:,11)';x(:,12)'],[x(:,13)';x(:,14)';x(:,15)';x(:,16)';x(:,17)'],[x(:,18)';x(:,19)';x(:,20)'],S,Cfixed), tol, 'y0', Cost_state, 'kickrank', 2);
        V_new = amen_cross_s({Cost_state_new}, @(x)logsmooth(x-tnew,eps_cvar), tol, 'y0', V);
        Vmean = dot(W,V_new);
        cv_new = tnew + Vmean / (1-beta_cvar);     
        Cost_new = cv_new + 0.5*eps*norm((unew.^2)*Wt);
        step = step/2;
    end
    if (Cost_new>Cost)
        cost_increase = cost_increase+1;
    end
    u = unew;
    t_cvar = tnew;    
    Cost_state = Cost_state_new;
    fprintf('Iter=%d, Cost=%g, cv=%g, t=%g, |G|=%g, step=%g\n', iter, Cost_new, cv_new, tnew, normGrad, step*2);

    figure(2); plot(Tnodes, u'); title('u'); legend('work', 'school', 'other'); drawnow;
    
    eps_cvar = max(eps_cvar*0.5, eps_final)

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
