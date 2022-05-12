function test_2d_elliptic_lagrangian
check_tt;
eps = parse_parameter('Smoothing parameter eps', 3e-3);
eps_factor = parse_parameter('eps damping factor', 0.7);
tol = parse_parameter('Approximation tolerance', 1e-3);
beta = parse_parameter('CVaR quantile beta', 0.8);
alpha = parse_parameter('Control regularization parameter alpha', 1e-4);


if (parse_parameter('Recompute random field?', false)) 
    model = Elliptic_Example_5_setup;   save('Ex5model.mat', 'model');
else
    load('Ex5model.mat');
end
d = numel(model.Axi)-1;
xif_grid = repmat({model.xif(:)}, d, 1);
Wf = mtkron(repmat({tt_tensor(model.wf)}, 1, d));

% Samples for control variate correction
XiSamples = [];


ttotal=tic;
[cv, u, t, ttranks] = ttrisk_lagrangian_uzawa(model, xif_grid, Wf, alpha,beta,eps, eps_factor, 200,tol, XiSamples);
toc(ttotal);


fprintf('CVaR:\t%g\n', cv);
fprintf('t:\t%g\n', t);
figure(1); plot_nodalfield(model, model.B*u, 'Control u (padded by zeros)');
figure(2); plot(ttranks); title('TT ranks'); xlabel('iteration'); legend('forward map', 'risk measures');


% Copy vars to main space
vars = whos;
for i=1:numel(vars)
    if (exist(vars(i).name, 'var'))
        assignin('base', vars(i).name, eval(vars(i).name));
    end
end
end


% 
% 
% 
% 
% % Forward solver (to replace with als_cross)
% function [dy] = forward_solve(model,y,u,xi)
% ny = size(model.A0,1);
% dy = zeros(size(y));
% for i=1:size(y,1)
%     A = model.A0;
%     for k=1:numel(model.A)
%         A = A + model.A{k}*xi(i,k);
%     end
%     b = model.B*u - A*y(i,:)';
%     dy(i,:) = A\b;  % Forward solution
% end
% end
% 
