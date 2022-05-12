function test_1d_elliptic(model)
check_tt;
ny = parse_parameter('(Odd) Number of spatial grid points n', 129);
eps = parse_parameter('Smoothing parameter eps', 3e-3);
eps_factor = parse_parameter('eps damping factor', 0.7);
tol = parse_parameter('Approximation tolerance', 1e-4);
beta = parse_parameter('CVaR quantile beta', 0.8);
alpha = parse_parameter('Control regularization parameter alpha', 1e-6);

d = parse_parameter('Number of random variables d', 5);

if (nargin<1) || (isempty(model))
    % Create a synthetic 1D Karhunen-Loeve expansion
    rng(5);
    sigma = 2^(0);
    nu = (ny-1)/2;
    % KL expansion
    x = (0.5:ny-1.5)'/(ny-1);  % Coeff is defined at midpoints
    w = ones(ny-1,1)/(ny-1);
    C = exp(-(x-x').^2/(2*0.25^2)); % Gaussian covariance
    WC = diag(sqrt(w))*C*diag(sqrt(w));
    WC = (WC+WC')*0.5;
    [Psi,L] = eig(WC);
    L = diag(L);
    [L,prm] = sort(L, 'descend');
    Psi = Psi(:,prm);
    Phi = sqrt(1./w).*Psi;
    model = struct();  % Simulate Ay = Bu, where A = A0 + xi_i A_i
    model.A0 = spdiags(ones(ny,1)*[-1 2 -1], -1:1, ny, ny)*(ny-1)^2;
    model.A0(1,2)=0; model.A0(2,1)=0; model.A0(ny-1,ny)=0; model.A0(ny,ny-1)=0;
    model.A0 = (model.A0 + model.A0')*0.5;
    model.A0 = model.A0 * 10; % mean field
    for i=1:d
        c = sigma*sqrt(L(i))*Phi(:,i);
        model.A{i} = sparse(ny,ny);  % midpoint rule for coeff integration
        model.A{i}(2:ny-1,2:ny-1) = spdiags([-c(2:end), c(1:end-1)+c(2:end), -c(1:end-1)]*(ny-1)^2, -1:1, ny-2, ny-2);
    end
    w = ones(ny,1); w(1)=0.5; w(end)=0.5; w = w/(ny-1); % trapezoidal rule for y
    model.My = spdiags(w,0,ny,ny); % Mass in state
    model.Mu = speye(nu,nu)/nu; % u is defined at midpoints
    model.B = sparse(ny,nu);
    model.B((ny-1)/2-nu/2+1:(ny-1)/2+nu/2+1, :) = spdiags(ones(nu+1,1)*[0.5 0.5], -1:0, nu+1, nu);
    model.yd = ones(ny,1); % desired state
end

% Grid in RVs
d = numel(model.A);
nxif = parse_parameter('Number of grid points in random variables', 17);  % Fine grid
[xif, wf] = lgwt(nxif,-sqrt(3),sqrt(3));  wf = wf/(2*sqrt(3)); % should integrate to 1
[xic,  ~] = lgwt(5,-sqrt(3),sqrt(3));  % Should suffice for forward solution only
xif_grid = repmat({xif(:)}, d, 1);
xic_grid = repmat({xic(:)}, d, 1);
Wf = mtkron(repmat({tt_tensor(wf)}, 1, d));

% % Function for control variate correction
% XiSamplesFun = @(xi)(2*xi-1)*sqrt(3);
XiSamplesFun = [];


tic;
[cv, u, t, ttranks, evalcnt, cgiters] = ttrisk(model, @g_grad_fun_1d, xic_grid, xif_grid, Wf, alpha,beta,eps,200,tol, false, eps_factor, XiSamplesFun);
toc;

fprintf('CVaR:\t%g\n', cv);
fprintf('t:\t%g\n', t);
xu = (((ny-1)/2 - nu/2+1 : (ny-1)/2 + nu/2)' + 0.5)/(ny-1);  % grid points for u
figure(1); plot(xu, u); title('Control u'); xlabel('x');
figure(2); plot(ttranks); title('TT ranks'); xlabel('iteration'); legend('forward map', 'risk measures');


% Copy vars to main space
vars = whos;
for i=1:numel(vars)
    if (exist(vars(i).name, 'var'))
        assignin('base', vars(i).name, eval(vars(i).name));
    end
end
end
