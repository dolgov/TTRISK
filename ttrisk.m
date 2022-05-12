function [cv, u, t, ttranks, evalcnt, cgiters] = ttrisk(model, g_grad_fun, xic_grid, xif_grid, Wf, alpha,beta,eps_final,maxiter,tol, fullHessian, eps_factor, XiSamplesFun)
% TTRISK optimization algorithm in the reduced space formulation
% Inputs:
%   model: a structure defining the model. Needs to have the following:
%          model.A0: mean-field matrix
%          model.A:  a cell array of affine field correction matrices
%          model.B:  control-to-state actuator matrix
%          model.My: state mass matrix
%          model.Mu: control mass matrix
%          model.yd: desired state
%   g_grad_fun: a function handle to solve the forward model (see e.g. g_grad_fun_1d.m)
%   xic_grid: a cell array of d vectors of coarse grid points in random variables
%   xif_grid: a cell array of d vectors of fine grid points in random variables
%   Wf: a tt_tensor of quadrature weights on the fine grid
%   alpha: control regularization parameter
%   beta: CVaR quantile parameter
%   eps_final: smoothing parameter eps in the end of iterations
%   maxiter: maximum number of iterations
%   tol: stopping and TT approximation tolerance
%   fullHessian: if true, assemble the full Hessian, otherwise use CG with
%                MatVec with a deterministic Hessian anchored at \xi_*
%   eps_factor: rate of decreasing of eps, i.e. eps_next = eps_prev * eps_factor
%   XiSamplesFun: if nonempty, a function handle that maps uniform(0,1)
%                 samples to the desired probability space of xi for
%                 control variate correction
%
% Outputs:
%   cv: CVaR estimate value
%   u: optimized control
%   t: optimized t value
%   ttranks: a vector of maximal TT ranks from all Newton iterations
%   evalcnt: a vector of numbers of forward model evaluations from all iterations
%   cgiters: a vector of CG iterations from all Newton iterations

nu = size(model.B,2);
Xic_grid_tt = tt_meshgrid_vert(cellfun(@tt_tensor, xic_grid, 'uni', 0));
d = Wf.d;
% Interpolate a Coarse grid (for forward model) to fine grid (for expectations)
CoarseToFine = arrayfun(@(i)reshape(lagrange_interpolant(xic_grid{i}, xif_grid{i}),1,numel(xif_grid{i}),numel(xic_grid{i})), (1:d)', 'uni', false);
CoarseToFine = cell2core(tt_matrix, CoarseToFine);

Nreplicas = 16;  % For control variate correction

% Initial guess
u = zeros(nu, 1);
Vg = 4; Vh = 4;
g = 4; grad_g = 4; hess_g = 4;

% Construct forward solution surrogate on a coarse grid first - it should
% be sufficient, since the solution is smoother than the indicator
     [g,~,~,~,eval0] = amen_cross_s(Xic_grid_tt, @(x)g_grad_fun(model, u, x, 0), tol, 'exitdir', 1, 'y0', g, 'dir', -1);
[grad_g,~,~,~,eval1] = amen_cross_s(Xic_grid_tt, @(x)g_grad_fun(model, u, x, 1), tol, 'kickrank', 0, 'exitdir', 1, 'y0', grad_g, 'dir', -1);
gf = CoarseToFine*g; % Interpolate onto fine grid for CVaR
grad_g_f = CoarseToFine*grad_g;

ttranks(1,1) = max(max(g.r), max(grad_g.r(1:d)));
evalcnt(1,1) = sum(eval0) + sum(eval1);

% Initial guess of t is important
t = dot(Wf,gf);
eps = max(abs(t)/2, eps_final) / eps_final;
eps = round(log2(eps)/(-log2(eps_factor)));
eps = (eps_factor^(-eps))*eps_final
t = 0;

tol_coeff = tol*0.1;

% Derivatives of the smoother at the given g
Vg = amen_cross_s({gf}, @(x)grad_logsmooth(x-t,eps), tol_coeff, 'y0', Vg, 'verb', 1);
Vh = amen_cross_s({gf}, @(x)hess_logsmooth(x-t,eps), tol_coeff, 'y0', Vh, 'verb', 1);

ttranks(1,2) = max(max(Vg.r), max(Vh.r));

grad_J_u = dot(grad_g_f, Wf.*Vg) / (1-beta) + alpha*model.Mu*u; % Assume L2 regularisation
grad_J_t = 1 - dot(Wf,Vg) / (1-beta);

cgiters = nan;  % This will store the number of CG iterations

for i=1:maxiter
    % Solve the newton step
    resid = [grad_J_u; grad_J_t];
    
    Vgmean = dot(Wf,Vg)/(1-beta);
    Vhmean = dot(Wf,Vh)/(1-beta);
    
    ttranks(i,:) = 0;
    evalcnt(i,1:2) = 0;
        
    % Hessian parts    
    % Anchor point for the single-point Hessian approximation
    xi_mean = cellfun(@(xi)dot(Wf.*(CoarseToFine*xi), Vg)/dot(Wf,Vg), Xic_grid_tt)        
    grad_g_loc = g_grad_fun(model, u, xi_mean, 1);
    grad_g_loc = grad_g_loc(:);

    Hess_J_ut = -dot(Wf.*Vh, grad_g_f) / (1-beta);
    
    if (fullHessian)
        [hess_g,~,~,~,eval2] = amen_cross_s(Xic_grid_tt, @(x)g_grad_fun(model, u, x, 2), tol, 'kickrank', 0, 'exitdir', 1, 'y0', hess_g, 'dir', -1);
        
        ttranks(i,1) = max(ttranks(i,1), max(hess_g.r(1:d)));
        evalcnt(i,1) = evalcnt(i,1) + sum(eval2);
        
        Hess_J_uu = reshape(dot(Wf.*Vg,CoarseToFine*hess_g), nu, nu) / (1-beta) + dot(Wf.*grad_g_f, Vh.*grad_g_f) / (1-beta) + alpha*model.Mu;
        
        Hess = [Hess_J_uu, Hess_J_ut'; Hess_J_ut, Vhmean];
        
        dsol = Hess\resid;
    else
        [dsol,~,~,cgiters(i)] = bicgstab(@(v)[g_grad_fun(model, u, xi_mean, 2, v(1:nu))'*Vgmean + grad_g_loc*(grad_g_loc'*v(1:nu))*Vhmean + alpha*model.Mu*v(1:nu) + Hess_J_ut'*v(nu+1); Hess_J_ut*v(1:nu) + Vhmean*v(nu+1)], ...
            resid, tol_coeff, 100);
    end
    
    % Line search
    step = 1;
    res_new = inf;
    Vhmean = 0;
    while ((norm(res_new)>norm(resid)) || (Vhmean<1e-2/eps)) && (step>1e-1)
        unew = u - step * dsol(1:nu);
        tnew = t - step * dsol(nu+1);
        
        gnew = g; grad_g_new = grad_g;
        % Wrap into exception handler, since too large steps may make the
        % solution uncomputable
        try
                 [gnew,~,~,~,eval0] = amen_cross_s(Xic_grid_tt, @(x)g_grad_fun(model, unew, x, 0), tol, 'exitdir', 1, 'dir', -1, 'y0', g);
            [grad_g_new,~,~,~,eval1]= amen_cross_s(Xic_grid_tt, @(x)g_grad_fun(model, unew, x, 1), tol, 'kickrank', 0, 'exitdir', 1, 'dir', -1, 'y0', grad_g);
            
            ttranks(i,1) = max([ttranks(i,1), max(gnew.r), max(grad_g_new.r(1:d))]);
            evalcnt(i,1) = evalcnt(i,1) + sum(eval0 + eval1);
        catch ME
        end
        gf = CoarseToFine*gnew; % Interpolate onto fine grid for CVaR
        grad_g_f = CoarseToFine*grad_g_new;

        % Derivatives of the smoother at the given g
        Vgnew = Vg;
        try
            Vgnew = amen_cross_s({gf}, @(x)grad_logsmooth(x-tnew,eps), tol_coeff, 'y0', Vg, 'verb', 1, 'nswp', 7);  % 16
            ttranks(i,2) = max(ttranks(i,2), max(Vgnew.r));
        catch ME
        end
        Vhnew = Vh;
        try
            Vhnew = amen_cross_s({gf}, @(x)hess_logsmooth(x-tnew,eps), tol_coeff, 'y0', Vh, 'verb', 1, 'nswp', 7);
            ttranks(i,2) = max(ttranks(i,2), max(Vhnew.r));
        catch ME
        end
                
        grad_J_u = dot(grad_g_f, Wf.*Vgnew) / (1-beta) + alpha*model.Mu*unew; % Assume L2 regularisation
        grad_J_t = 1 - dot(Wf,Vgnew) / (1-beta);
        
        % Control Variate correction
        if (~isempty(XiSamplesFun))&&(abs(eps/eps_final-1)<1e-10)
            % Sample initially the number of points equal to TT budget
            Nsamples = Nreplicas*round(sum(eval0 + eval1)/Nreplicas);
            XiSamples = XiSamplesFun(rand(Nsamples,d));
            g_MC = g_grad_fun(model, unew, XiSamples, 0);
            grad_g_MC = g_grad_fun(model, unew, XiSamples, 1);
            Vg_MC = tt_sample_lagr(Vgnew, xif_grid, XiSamples);
            gu_corr = (double(g_MC>tnew).*grad_g_MC - tt_sample_lagr(grad_g_f, xif_grid, XiSamples).*Vg_MC)/(1-beta);
            gu_corr = reshape(gu_corr, [], Nreplicas, nu);
            gu_corr = reshape(mean(gu_corr), Nreplicas, nu);
            std_u = norm(std(gu_corr)')
            gt_corr = (double(g_MC>tnew) - Vg_MC)/(1-beta);
            gt_corr = reshape(gt_corr, [], Nreplicas);
            gt_corr = mean(gt_corr)';
            std_t = std(gt_corr)
            % Sample more if we can't reach the threshold
            Nsamples = Nsamples/Nreplicas * max((norm([std_u; std_t])/tol).^2, 1)
            Nsamples = Nreplicas*round(Nsamples/Nreplicas);
            evalcnt(i,2) = evalcnt(i,2) + Nsamples;
            XiSamples = XiSamplesFun(rand(Nsamples,d));
            g_MC = g_grad_fun(model, unew, XiSamples, 0);
            grad_g_MC = g_grad_fun(model, unew, XiSamples, 1);
            Vg_MC = tt_sample_lagr(Vgnew, xif_grid, XiSamples);
            gu_corr = (double(g_MC>tnew).*grad_g_MC - tt_sample_lagr(grad_g_f, xif_grid, XiSamples).*Vg_MC)/(1-beta);
            gu_corr = reshape(gu_corr, [], Nreplicas, nu);
            gu_corr = reshape(mean(gu_corr), Nreplicas, nu);
            std_u = norm(std(gu_corr)')
            gt_corr = (double(g_MC>tnew) - Vg_MC)/(1-beta);
            gt_corr = reshape(gt_corr, [], Nreplicas);
            gt_corr = mean(gt_corr)';
            std_t = std(gt_corr)
            gu_corr = mean(gu_corr)';
            gt_corr = mean(gt_corr)
            grad_J_u = grad_J_u + gu_corr;
            grad_J_t = grad_J_t + gt_corr;
        end
                
        res_new = [grad_J_u; grad_J_t];
        
        Vhmean = dot(Wf,Vhnew)/(1-beta);
        
        step = step*0.5;
    end
    u = unew;
    t = tnew;
    Vh = Vhnew;
    Vg = Vgnew;
    g = gnew;
    grad_g = grad_g_new;

    if (fullHessian)
        fprintf('TTRISK: iter=%d, resid_u=%3.3e, resid_t=%3.3e, tnew=%g, step=%g, cond_u=%3.3e, H_tt=%3.3e\n\n', i, norm(grad_J_u), norm(grad_J_t), t, step*2, cond(Hess_J_uu), Vhmean);
    else
        fprintf('TTRISK: iter=%d, resid_u=%3.3e, resid_t=%3.3e, tnew=%g, step=%g, H_tt=%3.3e\n\n', i, norm(grad_J_u), norm(grad_J_t), t, step*2, Vhmean);
    end
    if (norm(res_new)<tol)&&(i>1)&&(abs(eps/eps_final-1)<1e-10)
        break;
    end
    % Reduce eps
    eps = max(eps*eps_factor, eps_final)
end

V = amen_cross_s({gf}, @(x)logsmooth(x-t,eps), tol_coeff, 'kickrank', 0.3);
cv = t + dot(Wf,V) / (1-beta);
if (~isempty(XiSamplesFun))
    cv_corr = mean(double(g_MC>t).*(g_MC-t) - tt_sample_lagr(V, xif_grid, XiSamples))/(1-beta)
    cv = cv + cv_corr;
end
end

