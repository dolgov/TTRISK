function [cv, u, t, ttranks] = ttrisk_lagrangian_uzawa(model, xi_grid, W, alpha,beta,eps_final, eps_factor, maxiter,tol, XiSamples)
% TTRISK optimization algorithm in the Lagrangian formulation
% Inputs:
%   model: a structure defining the model. Needs to have the following:
%          model.A0: mean-field matrix
%          model.A:  a cell array of affine field correction matrices
%          model.B:  control-to-state actuator matrix
%          model.My: state mass matrix
%          model.Mu: control mass matrix
%          model.yd: desired state
%   xi_grid: a cell array of d vectors of grid points in random variables
%   W: a tt_tensor of quadrature weights on the random variable grid
%   alpha: control regularization parameter
%   beta: CVaR quantile parameter
%   eps_final: smoothing parameter eps in the end of iterations
%   eps_factor: rate of decreasing of eps, i.e. eps_next = eps_prev * eps_factor
%   maxiter: maximum number of iterations
%   tol: stopping and TT approximation tolerance
%   XiSamples: if nonempty, random samples of xi for control variate correction
%
% Outputs:
%   cv: CVaR estimate value
%   u: optimized control
%   t: optimized t value
%   ttranks: a vector of maximal TT ranks from all Newton iterations


[ny,nu] = size(model.B);
Xi_grid_tt = tt_meshgrid_vert(cellfun(@tt_tensor, xi_grid, 'uni', 0));
d = W.d;
% sqrt of global quadrature weights for computing the norms
sqWspace = [{reshape(sqrt(spdiags(model.My,0)),1,ny)}; cellfun(@sqrt, core2cell(W), 'uni', 0)];
sqWspace = cell2core(tt_tensor, sqWspace);
nxi = Xi_grid_tt{1}.n;

Nsamples = size(XiSamples,1);

% Initial guess
u = zeros(nu, 1);
Y = tt_reshape(tt_zeros([ny; nxi]), nxi, [], ny, 1);
P = 0*Y;
% Initial guess of t is important
G = amen_cross_s({Y}, @(y)state_cost(model,y), tol, 'exitdir', -1);
Psi = eye(1,ny+1)*G;
gradPsi = [zeros(ny,1) eye(ny)]*G;

% Initial guess of t is important
t = dot(W,Psi);
eps = max(abs(t)/2, eps_final) / eps_final;
eps = round(log2(eps)/(-log2(eps_factor)));
eps = (eps_factor^(-eps))*eps_final
t = 0;

% Eliminate control from the Hessian explicitly
Mupi = spdiags(1./(sum(model.Mu,2)*alpha), 0, size(model.Mu,1), size(model.Mu,2));
BMB = sparse(model.B)*Mupi*sparse(model.B');

tol_coeff = tol*0.1;

gradV = 4; hessV = 4; Hyt = 4; gradPsiSq = 4;

for iter=1:maxiter        
    % Derivatives of the smoother at the given g
    gradV = amen_cross_s({Psi}, @(x)grad_logsmooth(x-t,eps)/(1-beta), tol_coeff, 'y0', gradV, 'verb', 1);    
    ttranks(iter,1) = max(gradV.r);
    hessV = amen_cross_s({Psi}, @(x)hess_logsmooth(x-t,eps)/(1-beta), tol_coeff, 'y0', hessV, 'verb', 1);
    ttranks(iter,1) = max(max(hessV.r), ttranks(iter,1));
    
    [gradPsiSq,~,~,ind] = amen_cross_s({tkron(tt_ones(ny), hessV), tt_reshape(gradPsi, [ny; nxi])}, @(x)x(:,1).*(x(:,2:end).^2), tol_coeff, 'exitdir', -1, 'dir', 1, 'lm', 'y0', gradPsiSq);
    gradPsi1 = tt_sample_ind(gradPsi, ind{2,1}).'; % size ny x r
    % Diagonal in r
    gradPsiD = zeros(ny, size(gradPsi1,2), size(gradPsi1,2));
    for i=1:size(gradPsi1,2)
        gradPsiD(:,i,i) = gradPsi1(:,i);
    end
    gradPsiD = reshape(gradPsiD, ny*size(gradPsi1,2), size(gradPsi1,2));
    gradPsi1 = lrmatrix(tt_sample_ind(hessV, ind{2,1}).' .* gradPsi1,  sparse(gradPsiD));
    
    % Hessian terms with t
    Hyt = amen_cross_s({Y}, @(y)Hyt_fun(model,y,t,beta,eps), tol_coeff, 'exitdir', -1, 'y0', Hyt, 'dir', 1); 
    Htt = dot(W,hessV);
    
    % Matrix
    Ablock = cell(3,3);
    %
    expander = [{sparse(model.My)}; core2cell(diag(gradV))];
    expander(:,2) = core2cell(gradPsiSq);
    expander{1,2} = gradPsi1;
    for i=1:d
        expander{i+1,2} = reshape(expander{i+1,2}, size(expander{i+1,2},1), nxi(i), 1, []);
        expander{i+1,2} = repmat(expander{i+1,2}, 1, 1, nxi(i), 1);
        for k=1:size(expander{i+1,2},4)
            for j=1:size(expander{i+1,2},1)
                expander{i+1,2}(j,:,:,k) = diag(expander{i+1,2}(j,:,1,k));
            end
        end
        expander{i+1,2} = reshape(expander{i+1,2}, size(expander{i+1,2},1)*nxi(i), []);
        expander{i+1,2} = sparse(expander{i+1,2});
        expander{i+1,1} = reshape(expander{i+1,1}, size(expander{i+1,1},1)*nxi(i), []);
        expander{i+1,1} = sparse(expander{i+1,1});
    end
    Ablock{1,1} = expander;
    %        
    Ablock{2,1} = model.Axi;  
    Ablock{1,2} = model.AxiAdj;
    %
    expander = core2cell(tt_reshape(Hyt, [ny; nxi], tol));
    expander{1} = reshape(expander{1}, ny, []); % n x r
    expander{1} = sparse(expander{1});
    expander{1} = [expander{1}; sparse(ny*(ny-1), size(expander{1},2))]; % size n*n' x r
    expander{1} = reshape(expander{1}, ny, []); % n x n'*r
    for i=1:d
        expander{i+1} = reshape(expander{i+1}, size(expander{i+1},1), nxi(i), 1, []);
        expander{i+1}(:,:,2:nxi(i),:) = 0;
        expander{i+1} = reshape(expander{i+1}, size(expander{i+1},1)*nxi(i), []);
        expander{i+1} = sparse(expander{i+1});
    end
    Ablock{1,3} = expander;
    %
    expander = core2cell(tt_reshape(Hyt, [ny; nxi], tol));
    expander{1} = reshape(expander{1}, 1, []); % n'*r
    expander{1} = sparse(expander{1});
    expander{1} = [expander{1}; sparse(ny-1, size(expander{1},2))]; % size n x n'* r
    for i=1:d
        expander{i+1} = expander{i+1} .* W{i};
        expander{i+1} = reshape(expander{i+1}, size(expander{i+1},1), 1, nxi(i), []);
        expander{i+1}(:,2:nxi(i),:,:) = 0;
        expander{i+1} = reshape(expander{i+1}, size(expander{i+1},1)*nxi(i), []);
        expander{i+1} = sparse(expander{i+1});        
    end
    Ablock{3,1} = expander;
    %
    Ablock{2,2} = [{-BMB}; arrayfun(@(i)lrmatrix(ones(nxi(i),1), W{i}'), (1:d)', 'uni', 0)];
    %
    Ablock{3,3} = [{Htt*speye(ny)}; arrayfun(@(n)speye(n), nxi(:), 'uni', 0)];
    
    % Residual
    pres = gradV.*gradPsi;
    pres = round(pres, tol_coeff);
    pres = tt_reshape(pres, [ny; nxi], tol_coeff);
    pres = pres + amen_mm(model.AxiAdj, tt_reshape(P, [ny; nxi], tol_coeff), tol_coeff);
    
    corr_t = 0;
    % MC correction
    if (Nsamples>0)&&(eps<3*eps_final)
        YSamples = tt_sample_lagr(Y, xi_grid, XiSamples);
        gradPsiSamples = state_cost(model, YSamples);
        PsiSamples = gradPsiSamples(:,1);
        gradPsiSamples = gradPsiSamples(:,2:end);
        GradVSamples = tt_sample_lagr(gradV, xi_grid, XiSamples);
        corr_t = double((PsiSamples - t)>0)/(1-beta) - GradVSamples;
        % Correction in P residual
        Corr_p = cell(d+1,1);
        for m=1:Nsamples
            Corr_p{1}(:,m) = gradPsiSamples(m,:)'*corr_t(m);
            % Sampled Lagrange polynomials
            for i=1:d
                Corr_p{i+1}(:,m) = lagrange_interpolant(xi_grid{i}, XiSamples(m,i))./W{i};
            end
        end
        Corr_p = amen_sum(Corr_p, ones(Nsamples,1), tol_coeff, 'can', true);
        Corr_p = Corr_p / Nsamples;
        pres = pres + Corr_p;
        figure(1);
        plot([GradVSamples - grad_logsmooth(PsiSamples-t, eps)/(1-beta)]);
        title('Error in grad V');
        figure(2);
        plot([double((PsiSamples - t)>0)/(1-beta) - GradVSamples])
        title('Error in Indicator');
        corr_t = mean(corr_t)
    end
        
    % control is now directly resolvable from Lagrange multipliers
    u = (model.Mu \ (model.B'*dot(P,W)) )/alpha;
    %
    yres = amen_mm(model.Axi, tt_reshape(Y, [ny; nxi], tol_coeff), tol_coeff) - tkron(tt_tensor(model.B*u), tt_ones(nxi));
    %
    tres = 1 - dot(W,gradV) - corr_t;
    tres = tkron(tt_tensor([tres; zeros(ny-1,1)]), tt_unit(nxi,d,ones(d,1)));
    
    resid = {pres; yres; tres};
    if (iter==1)
        resid0 = [norm(sqWspace.*pres); norm(sqWspace.*yres); norm(tres)];
                                                              % ^ single number   
    end
    
    % Solve!
    tol_solve = min(tol*norm(resid0)/norm([norm(sqWspace.*pres); norm(sqWspace.*yres); norm(tres)]), 1)*0.5
    X0 = [tt_unit([ny; nxi],d+1,1); tt_unit([ny; nxi],d+1,1); tt_unit([ny; nxi],d+1,1)];
    Soln = amen_block_solve(Ablock, resid, tol_solve, 'exitdir', -1, 'x0', X0, 'kickrank', 0, 'nswp', 5, 'trunc_norm', 'fro', ...
           'solfun', @(i, XAX1, Ai, XAX2, XY1, yi, XY2, tol, sol_prev)cvar_localsolve(i, XAX1, Ai, XAX2, XY1, yi, XY2, tol, sol_prev, nu));
    
    ttranks(iter,2) = max(Soln.r);   
       
    % Extract solution components
    dy = tt_reshape([1 0 0]*Soln, nxi, tol_coeff, ny, 1);
    dp = tt_reshape([0 1 0]*Soln, nxi, tol_coeff, ny, 1);
    dt = [0 0 1]*chunk(Soln,1,1)*dot(chunk(Soln,2,d+1), tt_unit(nxi,d,ones(d,1)));
    dt = full(dt);
    dt = dt(1);  
    
    % Line search
    resid = [norm(sqWspace.*pres); norm(sqWspace.*yres); norm(tres)];
    step = 1;
    res_new = inf;
    Htt = 0;
    fail = false;
    while ((norm(res_new)>norm(resid)) || (Htt<1e-2/eps) || (fail)) && (step>1e-2)
        Ynew = round(Y - step * dy, tol_coeff); 
        Pnew = round(P - step * dp, tol_coeff);
        tnew = t - step * dt;

        % Cost function
        G = amen_cross_s({Ynew}, @(y)state_cost(model,y), tol_coeff, 'exitdir', -1, 'y0', G, 'dir', 1);
        Psi = eye(1,ny+1)*G;
        gradPsi = [zeros(ny,1) eye(ny)]*G;
        % Derivatives of the smoother at the given g
        fail = false;
        try
            gradV = amen_cross_s({Psi}, @(x)grad_logsmooth(x-tnew,eps)/(1-beta), tol_coeff, 'y0', gradV, 'verb', 1);
            ttranks(iter,1) = max(max(gradV.r), ttranks(iter,1));                        
        catch ME
            fprintf('gradV failed with %s\n', ME.message);
            fail = true;
        end
        try 
            hessV = amen_cross_s({Psi}, @(x)hess_logsmooth(x-tnew,eps)/(1-beta), tol_coeff, 'y0', hessV, 'verb', 1);
            ttranks(iter,1) = max(max(hessV.r), ttranks(iter,1));
        catch ME
            fprintf('hessV failed with %s\n', ME.message);
            fail = true;
        end
        
        if (~fail)
            % New residuals
            pres = gradV.*gradPsi;
            pres = round(pres, tol_coeff);
            pres = tt_reshape(pres, [ny; nxi], tol_coeff);
            pres = pres + amen_mm(model.AxiAdj, tt_reshape(Pnew, [ny; nxi], tol_coeff), tol_coeff);
                                    
            corr_t = 0;
            if (Nsamples>0)&&(eps<3*eps_final)
                % MC correction
                YSamples = tt_sample_lagr(Ynew, xi_grid, XiSamples);                
                gradPsiSamples = state_cost(model, YSamples);
                PsiSamples = gradPsiSamples(:,1);
                gradPsiSamples = gradPsiSamples(:,2:end);
                GradVSamples = tt_sample_lagr(gradV, xi_grid, XiSamples);
                corr_t = double((PsiSamples - tnew)>0)/(1-beta) - GradVSamples;
                % Correction in P residual
                Corr_p = cell(d+1,1);
                for m=1:Nsamples
                    Corr_p{1}(:,m) = gradPsiSamples(m,:)'*corr_t(m);
                    % Sampled Lagrange polynomials
                    for i=1:d
                        Corr_p{i+1}(:,m) = lagrange_interpolant(xi_grid{i}, XiSamples(m,i))./W{i};
                    end
                end
                Corr_p = amen_sum(Corr_p, ones(Nsamples,1), tol_coeff, 'can', true);
                Corr_p = Corr_p / Nsamples;
                pres = pres + Corr_p;
                corr_t = mean(corr_t);
            end
            
            %
            unew = (model.Mu \ (model.B'*dot(Pnew,W)) )/alpha;
            %
            yres = amen_mm(model.Axi, tt_reshape(Ynew, [ny; nxi], tol_coeff), tol_coeff) - tkron(tt_tensor(model.B*unew), tt_ones(nxi));
            %
            tres = 1 - dot(W,gradV)  - corr_t;
            tres = tkron(tt_tensor([tres; zeros(ny-1,1)]), tt_unit(nxi,d,ones(d,1)));
            
            res_new = [norm(sqWspace.*pres); norm(sqWspace.*yres); norm(tres)];
            
            % New time hessian
            Htt = dot(W,hessV);
        end        
        
        step = step*0.5;
    end
    Y = Ynew;
    u = unew;
    P = Pnew;
    t = tnew;
    
    fprintf('ttrisk_lagr: iter=%d, res_y=%3.3e, res_p=%3.3e, res_t=%3.3e, tnew=%g, step=%g, Htt=%3.3e\n\n', iter, norm(sqWspace.*yres), norm(sqWspace.*pres), norm(tres), t, step*2, Htt);
    if (norm(tres)<tol)&&(iter>1)&&(abs(eps/eps_final-1)<1e-10)
        break;
    end
    % Tighten the smoothing factor
    eps = max(eps*eps_factor, eps_final)
end

V = amen_cross_s({Psi}, @(x)logsmooth(x-t,eps)/(1-beta), tol);
cv = t + dot(W,V);

if (Nsamples>0)
    % MC correction
    PsiSamples = tt_sample_lagr(Psi, xi_grid, XiSamples);
    corr_cv = double((PsiSamples - t)>0).*(PsiSamples - t)/(1-beta) - tt_sample_lagr(V, xi_grid, XiSamples);
    mean_cv = mean(corr_cv)
    
    cv = cv + mean_cv;
end

end




function [Hyt] = Hyt_fun(model, y, t,beta,eps)
Hyt = zeros(size(y));
for i=1:size(y,1)
    gradPsi = model.My*(y(i,:)' - model.yd); % ny x 1
    Psi = 0.5*(y(i,:)' - model.yd)'*gradPsi;
    Hyt(i,:) = -hess_logsmooth(Psi-t,eps)/(1-beta) * gradPsi';
end
end

% Objective function = QoI in *state* and its gradient
function [gradPsi] = state_cost(model, y)
gradPsi = model.My*(y' - model.yd); % ny x I
gradPsi = [zeros(size(gradPsi,2),1), gradPsi'];
for i=1:size(y,1)
    gradPsi(i,1) = 0.5*(y(i,:)' - model.yd)'*gradPsi(i,2:end)';
end
end

