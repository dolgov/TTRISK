function [G] = g_grad_fun_1d(model,u,xi, deriv, vec)
%%% Forward model solution function for toy 1D elliptic problem
% Inputs:
%   model: a structure defining the model. Needs to have the following:
%          model.A0: mean-field matrix
%          model.A:  a cell array of affine field correction matrices
%          model.B:  control-to-state actuator matrix
%          model.My: state mass matrix
%          model.Mu: control mass matrix
%          model.yd: desired state
%   u: control vector
%   xi: I x d matrix of samples of random variables
%   deriv: the degree of the derivative to compute:
%       0: misfit function 0.5 ||y-yd||_{My}^2
%       1: gradient_u of misfit
%       2: Hessian_{uu} of misfit
%   vec: if nonempty, and deriv=2, multiply the Hessian by vec instead of
%        assembling the entire Hessian
%
% Outputs:
%   G: I x N matrix of outputs, where
%       N = 1 for deriv = 0
%       N = n_u for deriv = 1
%       N = n_u^2 for deriv = 2 and empty or missing vec
%       N = n_u for deriv = 2 and nonempty vec
%
I = size(xi,1);
nu = size(model.B,2);
for i=I:-1:1
    A = model.A0;
    for k=1:numel(model.A)
        A = A + model.A{k}*xi(i,k);
    end
    b = model.B*u;
    y = A\b;  % Forward solution
    b = model.My*(y-model.yd);
    phi = (A')\b; % Adjoint (sensitivity) soln
    if (nargin<4)
        G(i,1:1+nu) = [0.5*(y-model.yd)'*b   phi'*model.B];
    else
        if (deriv==0)
            G(i,1) = 0.5*(y-model.yd)'*b;
        end
        if (deriv==1)
            G(i,1:nu) = phi'*model.B;
        end
        if (deriv==2)
            if (nargin>4)
                % Matvec only
                w = model.B*vec;
                w = A\w;
                w = model.My*w;
                w = (A')\w;
                w = model.B'*w;
                G(i,1:nu) = w;
            else
                % Full hessian
                G(i,1:nu^2) = full(reshape((A\model.B)'*model.My*(A\model.B), 1, nu^2)); % B'*A^{-T}*My*A^{-1}*B
            end
        end
    end
end
end

