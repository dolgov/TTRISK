function [model]=Elliptic_Example_5_setup()
%  An Introduction to Optimal Control of Partial Differential Equations under Uncertainty
%  
%  Example 5: Risk Averse Optimal control the random Poisson's equation (problem Pepsilon)
% 
% 
% Copyright (c) 2018 J. Martinez-Frutos, F. Periago
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without modification, 
% are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation 
% and/or other materials provided with the distribution.
% 
% 3. Neither the name of the copyright holder nor the names of its contributors
% may be used to endorse or promote products derived from this software 
%  without specific prior written permission.
%
% 4. In all cases, the software is, and all modifications and derivatives of the 
% software shall be, licensed to you solely for use in conjunction with MathWorks 
% products and service offerings.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
% OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
% OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
% OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
% OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
% OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% 

%  addpath('./sparse-grids-matlab-kit/src/');
%  addpath('./sparse-grids-matlab-kit/main');
%  addpath('./sparse-grids-matlab-kit/tools/idxset_functions/');
%  addpath('./sparse-grids-matlab-kit/tools/knots_functions/');
%  addpath('./sparse-grids-matlab-kit/tools/lev2knots_functions/');
%  addpath('./sparse-grids-matlab-kit/tools/polynomials_functions/');
%  addpath('./sparse-grids-matlab-kit/tools/rescaling_functions/');
%  addpath('./sparse-grids-matlab-kit/docs-examples/');


fprintf('%-41s\n','');
fprintf('%-41s\n','************************************************************');
fprintf('%-41s\n','                       Creating Finite Element Model                        ');
fprintf('%-41s\n','************************************************************');


% Import .msh file
file='mesh_coarse.msh';
mesh.N_n           = dlmread(file,'',[0 0 0 0]);
mesh.N_e           = dlmread(file,'',[0 1 0 1]);
mesh.N_b           = dlmread(file,'',[0 2 0 2]);
mesh.nodes       = dlmread(file,'',[1 0 mesh.N_n 2]);
mesh.elements  = dlmread(file,'',[mesh.N_n+1 0 mesh.N_n+mesh.N_e 3]);
mesh.boundary  = dlmread(file,'',[mesh.N_n+mesh.N_e+1 0 mesh.N_n+mesh.N_e+mesh.N_b 2]);
mesh.control=find((mesh.nodes(:,1)-0.5).^2+(mesh.nodes(:,2)-0.5).^2<(0.25+1e-5)^2);


% Create FE model
model.mesh=mesh;
model.a=@(x,y)1;
model.f=zeros(model.mesh.N_n,1);
model.f(model.mesh.control)=ones(length(model.mesh.control),1);

% Assembly FE model
[model]=Assemble_A(model);   % Assembly system matrix.
[model]=Assemble_M(model);  % Assembly mass matrix.
[model]=Assemble_F(model);   % Assembly mass matrix.
[model]=Dirichlet_BC(model);   % Impose Dirichlet BC.
% Solve system of FE equations.
[model]=Solve_FEM(model) ;  

% plot mesh
% plot_FEmodel(model);


model.yd=ones(numel(model.y), 1);       % target
model.epsilon=1e-4;     % probability threshold in the cost functional(eq. 104)
%%%%% Galerkin Finite Element Method  K*bn=lambda*M*bn
fprintf('%-41s\n','');
fprintf('%-41s\n','************************************************************');
fprintf('%-41s\n','Computing eigenvalues and eigenvectors using Galerkin method');
fprintf('%-41s\n','************************************************************');

model.random.a_sigma=0.05;                      % standard deviation of the RF % 0.1
model.random.a_mean=1;                       % mean value of the RF
model.random.xlower=[0 0];                   % lower bound of spatial domain
model.random.xupper=[1 1];                   % upper bound of spatial domain
model.random.Lc = [0.5 0.5];                 % correlation length
model.random.corr_func{1,1}  = @(x,y) exp(-(abs(x-y)/model.random. Lc(1)).^2); % covariance function
model.random.corr_func{2,1}  = @(x,y) exp(-(abs(x-y)/model.random. Lc(2)).^2); % covariance function
model.random.Nterms = parse_parameter('Number of terms of the K-L expansion', 10);

fprintf('%-41s','Solving eigenvalue problem...');
[model]=KL_eigen(model);     
fprintf('%-41s\n','OK');

for i=1:model.random.Nterms
fprintf('%-25s\n',['Eigenvalue ',num2str(i),'= ',num2str(model.random.sv(i,i)),';']);
fprintf('%-25s\n',['captured the ',num2str((sum(diag(model.random.sv(1:i,1:i)))/model.random.a_sigma^2)*100),'% of the RF energy']);
end
    
fprintf('%-41s\n',' ');
fprintf('%-41s','Random field realization, a ...');

amean = 0;
for i=1:1    
    xi=random('normal',0,1,model.random.Nterms,1);
    aKL=model.random.bases*sqrt(model.random.sv)*xi; % K-L expanssion
    
    % Lognormal transformation
    model.random.lambdaf = log((model.random.a_mean.^2)./sqrt(model.random.a_sigma.^2+model.random.a_mean.^2));
    model.random.epsilonf = sqrt(log(model.random.a_sigma.^2./(model.random.a_mean.^2)+1));
    aKL_log=exp(model.random.lambdaf+model.random.epsilonf.*aKL);
    amean = amean + aKL_log;    
end
amean = amean/i;
avar = 0;
for i=1:1    
    xi=random('normal',0,1,model.random.Nterms,1);
    aKL=model.random.bases*sqrt(model.random.sv)*xi; % K-L expanssion
    
    % Lognormal transformation
    model.random.lambdaf = log((model.random.a_mean.^2)./sqrt(model.random.a_sigma.^2+model.random.a_mean.^2));
    model.random.epsilonf = sqrt(log(model.random.a_sigma.^2./(model.random.a_mean.^2)+1));
    aKL_log=exp(model.random.lambdaf+model.random.epsilonf.*aKL);
    avar = avar + (aKL_log-amean).^2;    
end
avar = avar/(i-1);


% % plot random field realization
% plot_elemfield(model,aKL_log,'Random field realization');
% fprintf('%-41s\n','OK');



%% %%% Impose control on source term of PDE      
model.u=eye(length(model.mesh.control),1);       
     
model.f=zeros(model.mesh.N_n,1);  
model.f(model.mesh.control)=model.u;
[model]=Assemble_F(model);   % Assemble RHS.
        
% Create the actuator matrix (B in the notes)
Control_Actuator_Matrix = sparse(model.mesh.N_n, length(model.mesh.control));
Control_Actuator_Matrix(model.mesh.control, :) = speye(length(model.mesh.control));
Control_Actuator_Matrix = model.M * Control_Actuator_Matrix;  % [Mu; *]
b=model.mesh.boundary(model.mesh.boundary(:,3)<5,1:2);
b=unique(b);
Control_Actuator_Matrix(b, :) = 0;  % [Mu; 0]

% solve PDE imposing the random field realization and plot state variable
[model_adaptive]=Assemble_A(model,aKL_log);       % Impose random field on A.
[model_adaptive]=Dirichlet_BC(model_adaptive);    % Impose Dirichlet BC.
[model_adaptive]=Solve_FEM(model_adaptive) ;      % Solve system of equations.

%%% Solution of the random PDE
soln_rand= model_adaptive.y;

% QoI
dy = model_adaptive.y - model.yd;
g = 0.5 * dy'*model.M*dy; % G = 0.5||y-yd||^2

% plot_nodalfield(model,soln_rand,'Random realization of y obtained from PDE')

% Solve the (self-)adjoint problem on the sensitivity function
model.f = model.M * dy; % from G = 0.5||y-yd||^2
[model]=Assemble_F(model);
[model_adaptive]=Assemble_A(model,aKL_log);       % Impose random field on A.
[model_adaptive]=Dirichlet_BC(model_adaptive);    % Impose Dirichlet BC.
[model_adaptive]=Solve_FEM(model_adaptive) ;      % Solve system of equations.
Sensitivity_Solution = model_adaptive.y;

% Gradient of QoI in control
Grad_g = Sensitivity_Solution'*Control_Actuator_Matrix;

% Grid in random variables
nxi = 9;
[xi,w] = gauss_hermite_rule(nxi);
Xi = tt_meshgrid_vert(tt_tensor(xi), model.random.Nterms);

% TT of matrix nonzeros
tol = 1e-3;

[irow,jcol,v] = find(model_adaptive.Ab);
a = amen_cross_s(Xi, @(xi)Anonzeros(model,irow,jcol,xi'), tol, 'vec', false, 'kickrank', 0, 'exitdir', -1);

a = tt_reshape(a, [a.r(1); a.n], tol*1e-2);
a1 = squeeze(a{1});
A1 = [];
for k=1:size(a1,2)
    A1 = [A1, sparse(irow,jcol,a1(:,k),size(model_adaptive.Ab,1), size(model_adaptive.Ab,2))];
end
Att = core2cell(a);
Att{1} = A1;
% Diagonalize in random vars
for k=2:numel(Att)
    a1 = Att{k};
    Att{k} = zeros(size(a1,1), numel(xi), numel(xi), size(a1,3));
    for j=1:numel(xi)
        Att{k}(:,j,j,:) = a1(:,j,:);
    end
    Att{k} = reshape(Att{k}, size(a1,1)*numel(xi), []);
    Att{k} = sparse(Att{k});
end

model.Axi = Att;
model.AxiAdj = Att; % it's symmetric
model.B = Control_Actuator_Matrix;

model.My = model.M; % Mass in state
model.Mu = model.M(model.mesh.control, model.mesh.control); % Mass in control
model.xif = xi;
model.wf = w;
end



function [f,g]=funceval(x,model,pce,xpert)
nMC=size(xpert,1);
xx=ones(1,931);
xx(1,1:10)=x;
% Impose control on source term of PDE
        model.f=sparse(model.mesh.N_n,1);
        model.f(model.mesh.control)=xx';
        [model]=Assemble_F(model);   % Assembly RHS.

        % update PCE
        [pce]=fitPCE(model,pce);   % fit PCE 
        [S_MC]=evaluatePCE(pce,xpert,1); % Evaluate PCE at Monte Carlo samples

        % Compute cost functional (eq.79)
        J0=sparse(nMC,1);
        r=sparse(nMC,1);
        
        for ii=1:nMC
        r(ii)=intD(model,(S_MC(ii,1:model.mesh.N_n)'-model.yd).^2);
        end
        
        % Compute epsilon_k
        model.epsilon_k=max(model.epsilon,0);
        model.alpha=0.01*std(r);

        for ii=1:nMC
            J0(ii)=(1+exp((-2/model.alpha)*(r(ii)-model.epsilon_k)))^(-1);
        end
        f=mean(J0)
        
        % Compute gradient (eq.83)
           C=(4/model.alpha)*...
                        (exp((-2/model.alpha)*(r-model.epsilon_k))).*...
                         (1+exp((-2/model.alpha)*(r-model.epsilon_k))).^(-2); %  C(z) eq.82
                     
        while  sum(isnan(C))>0
            model.alpha=1.1*model.alpha;   % increase value of alpha
            C=(4/model.alpha)*...
                        (exp((-2/model.alpha)*(r-model.epsilon_k))).*...
                         (1+exp((-2/model.alpha)*(r-model.epsilon_k))).^(-2); %  C(z) eq.82
        end

        
        p=repmat(C,1,size(S_MC,2)/2).*S_MC(:,1+model.mesh.N_n:end);   % adjoint state eq.109
        J_prim=mean(p,1)';
        g=J_prim(model.mesh.control)'; % descent direction
        g=g(1:10);

end

% FEM functions
function [model]=Assemble_M(model)
[model.M]=mass_matrixP1_2D(model.mesh.elements(:,1:3),model.mesh.areas,model.mesh.nodes(:,1:2));
end



function [out]=intD(model,in)
   out=in'*model.M*ones(model.mesh.N_n,1);
end

% Plot functions
function []=plot_mesh(model)
patch('vertices',model.mesh.nodes,'faces',model.mesh.elements(:,1:3),'edgecol','k','facecol',[.8,.9,1]);
axis square
end
function []=plot_elemfield(model,field,tit)
coord=zeros(2,3,model.mesh.N_e);
for d=1:2
    for i=1:3
        coord(d,i,:)=model.mesh.nodes(model.mesh.elements(:,i),d);
    end
end   
figure;
fill(squeeze( coord(1,:,:)),squeeze( coord(2,:,:)),field','edgecol','none');
set(gcf,'Name',tit,'NumberTitle','off','Toolbar','none');
set(gcf,'Visible','on');  
set(gcf,'Color',[1 1 1]);
view(2)
xlabel('x');  ylabel('y');
colorbar('location','EastOutside');
grid off;  axis square
title(tit);

end
function []=plot_nodalfield(model, field,tit)

 figure;
trisurf(model.mesh.elements(:,1:3),model.mesh.nodes(:,1),model.mesh.nodes(:,2),full(field),'edgecolor','none','facecolor','interp');
set(gcf,'Name',tit,'NumberTitle','off','Toolbar','none');
set(gcf,'Visible','on');  
set(gcf,'Color',[1 1 1]);
view(2)
xlabel('x');  ylabel('y');
colorbar('location','EastOutside');
grid off;  axis square
title(tit);

end


% Karhunen-Loeve expansion of random field
function [model]=KL_eigen(model)


xlower=model.random. xlower;                    % lower bound of spatial domain
xupper=model.random. xupper;                    % upper bound of spatial domain
Lc=model.random. Lc;                    % correlation length
corr_func=model.random. corr_func; % covariance function
Nterms=model.random. Nterms;               % Number of terms of the K-L expanssion
eigvec1d_galerkin=cell(2,1);
eigval1d_galerkin=cell(2,1);
nodes=cell(2,1);
for i=1:length(Lc)
numelem=100;
numnodes=numelem+1;
numdofs=numnodes;
nodes{i} = linspace(xlower(i),xupper(i),numnodes);
elem= [(1:numnodes-1)' ,(2:numnodes)' ];
gp_x=[-0.774596669241483 0 0.774596669241483];
gp_w=[0.555555555555556 0.888888888888889 0.555555555555556];
N = @(x)[ -0.5*(x-1) 0.5*(x+1) ];   % Lagrangian shape functions
              
% Computing matrix M    Eq. (26)   
Me     = zeros(2,2);
det_Je = 0.5*((xupper(i)-xlower(i))/numelem);
for j = 1:length(gp_x)
      NN    = N(gp_x(j));    
      Me = Me + NN'*NN*det_Je*gp_w(j);
end
% Assembly       
tripi=reshape(repmat(elem,1,2)',numelem*4,1);
tripj=reshape([elem(:,1) elem(:,1) elem(:,2) elem(:,2)]',numelem*4,1);
tripk=repmat(Me(:),numelem,1);
M = sparse(tripi,tripj,tripk);

% Computing matrix K     Eq. (26) 
K = sparse(numdofs,numdofs);
for e1 = 1:numelem
    xe1=nodes{i}(elem(e1,:));
   for e2 = 1:numelem
      Ke    = zeros(2);
      xe2=nodes{i}(elem(e2,:));
      for gp1 = 1:3
         NNe1    = N(gp_x(gp1));    
         xgp1     = NNe1*xe1';      
         for gp2 = 1:3
            NNe2     = N(gp_x(gp2));   
            xgp2     = NNe2*xe2';         
            Ke = Ke + corr_func{i,1}(xgp1,xgp2)*NNe1'*NNe2*det_Je*det_Je*gp_w(gp1)*gp_w(gp2);
         end
      end      
       K(elem(e1,:),elem(e2,:)) = K(elem(e1,:),elem(e2,:)) + Ke;
   end
end


[eigvec1d_galerkin{i},eigval1d_galerkin{i}]        = eigs(K,M,Nterms);  
[eigval1d_galerkin{i},index] = sort(diag(eigval1d_galerkin{i}),'descend');
eigvec1d_galerkin{i}       = eigvec1d_galerkin{i}(:,index);
end
% Eigenvalues 
eigindex=fliplr(fullfact([Nterms Nterms]));
eigval_galerkin=eigval1d_galerkin{1}(eigindex(:,1)).*eigval1d_galerkin{2}(eigindex(:,2));
[eigval_galerkin,I]=sort(eigval_galerkin,'descend');
eigindex=eigindex(I,:);
eigval_galerkin=eigval_galerkin(1:Nterms);
eigindex=eigindex(1:Nterms,:);

% Eigenfunctions
eigfun_galerkin= cell(Nterms,1);
for j = 1:Nterms  
eigfun_galerkin{j} = @(x1,x2) interp1(nodes{1},eigvec1d_galerkin{1}(:,eigindex(j,1)),x1,'pchip').*...
    interp1(nodes{2},eigvec1d_galerkin{2}(:,eigindex(j,2) ),x2,'pchip');
end

X=model.mesh.qpt(:,1);
Y=model.mesh.qpt(:,2);
model.random.bases=zeros(model.mesh.N_e,Nterms);
for j=1:Nterms
    model.random.bases(:,j)=eigfun_galerkin{j}(X,Y);
end  
model.random.sv=diag(eigval_galerkin);
end

% PCE expansion
function [pce]=fitPCE(model,pce,compute_adjoint)
%     Polynomial Chaos Expansion of y 
%     number of basis vectors Psi in the Polynomial chaos associated with 
%      (M , p_order) 
if nargin<3
    compute_adjoint=1;
end
P = sum(factorial(pce.M+[0:1:pce.p_order]-1)./(factorial([0:1:pce.p_order]).*factorial(pce.M-1)));

% Calculate 1D polynomials
pce.Psi=cell(pce.p_order+1,pce.M);
for i=1:pce.M
pce.Psi(1:pce.p_order+1,i) =Hermite(pce.p_order);
end  

% M-dimensional PC computation
        switch pce.anisotropic_pc
            case(0)
               pce.alpha  = multi_index(pce.M,pce.p_order);  % create the multi-index
               pce.P=P;
            case(1)
        
               rule=@(I) rulew(I,pce.Quadrature.g);
               pce.N=length(pce.Quadrature.g);
               pce.alpha=multiidx_gen(pce.N,rule,pce.p_order,1);
               pce.alpha=pce.alpha-1;     
               pce.P=size(pce.alpha,1);
        end
        
pce.PsiSqNorm  = prod(factorial(pce.alpha),2); % Calculate the square norm

% Find the PC coefficients using proyection approach using sparse grids
rule=@(I) rulew(I,pce.Quadrature.g);
N=length(pce.Quadrature.g);
Indexes=multiidx_gen(N,rule,pce.Quadrature.level,1);
[Afull,~] = smolyak_grid_multiidx_set(Indexes,{@(x)knots_gaussian(x,0,1)},@lev2knots_lin);
A=reduce_sparse_grid(Afull); % Remove repeated nodes
pce.Quadrature.xk=A.knots';       % sparse grid nodes (normalized random space)
pce.Quadrature.wk=A.weights';   % sparse grid weights (normalized random space)

pce.Quadrature.yk=zeros(size(pce.Quadrature.xk,1) ,pce.S);

    for ik=1:size(pce.Quadrature.xk,1)   % loop over the sparse grid nodes
    aKL=model.random.bases*sqrt(model.random.sv)*pce.Quadrature.xk(ik,:)';   % KL expasion
    aKL_log=exp(model.random.lambdaf+model.random.epsilonf.*aKL); % Lognormal transformation
    
    % Compute state equation at sparse grid nodes
    [model_stateeq]=Assemble_A(model,aKL_log);   % Impose random field on A.
    [model_stateeq]=Dirichlet_BC(model_stateeq); % Impose Dirichlet BC.
    [model_stateeq]=Solve_FEM(model_stateeq) ;  % Solve system of equations.
     pce.Quadrature.yk(ik,1:model.mesh.N_n)=model_stateeq.y';  % solution of direct problem
    
     if compute_adjoint==1
     % Compute adjoint state equation at sparse grid nodes
     model_adjointeq=model_stateeq;        
     model_adjointeq.f=(model_stateeq.y-model.yd);   
     model_adjointeq=Assemble_F(model_adjointeq);   % Assembly RHS.¡
     model_adjointeq=Dirichlet_BC(model_adjointeq); % Impose Dirichlet BC.
     model_adjointeq=Solve_FEM(model_adjointeq) ;  % Solve system of equations.    
     pce.Quadrature.yk(ik,model.mesh.N_n+1:end)=model_adjointeq.y';  % solution of direct problem
     end
    end    
   

    

pce.Ypsi=Psi_solve(pce,pce.Quadrature.xk);  % evaluate polynomials at stochastic nodes

% PCE coefficients
pce.up = zeros(pce.P,pce.S);
for i=1:pce.S
    pce.up(:,i)= (pce.Quadrature.wk'*(repmat(pce.Quadrature.yk(:,i),1,pce.P).*pce.Ypsi))';
end
 pce.up = pce.up./repmat(pce.PsiSqNorm,1,pce.S);


end
function [Y]=evaluatePCE(pce,X,folds,ny)
       % [Y,varargout]=solve(obj,X,ny,folds)
        %
        % Description: Generate an array containing the pce prediction at sites X. 
        %
        %
        % Inputs:
        %         X:           sample sites
        %         ny:          response to be obtained 
        
         if ~exist('folds','var')
              folds=1;  
         end
       

        sizeX=size(X,1);
        quo=floor(sizeX/folds); 

if folds==1
            Ypsi=Psi_solve(pce,X);  % evaluate polynomials at  sample sites
            Ytot=Ypsi*pce.up;
                if exist('ny','var')&&~isempty(ny)
                Y=Ytot(:,ny);  
                else
                Y=Ytot;
                end
%                        if  obj.trunc_negative==1
%                           Y(Y<0)=0;
%                         end
else
            for ii=1:folds-1
                Ypsi{ii,1}=Psi_solve(pce,X((ii-1)*quo+1:(ii)*quo,:));  % evaluate polynomials at stochstic nodes
                Ytot{ii,1}=Ypsi{ii,1}*pce.up;
                if exist('ny','var')&&~isempty(ny)
                Ycell{ii,1}=Ytot{ii,1}(:,ny);  
                else
                Ycell{ii,1}=Ytot{ii,1};
                end
%                         if  obj.trunc_negative==1
%                           Y{ii,1}(Y{ii,1}<0)=0;
%                         end
            end
                Ypsi{folds,1}=Psi_solve(pce,X((folds-1)*quo+1:end,:));  % evaluate polynomials at stochstic nodes
                Ytot{folds,1}=Ypsi{folds,1}*pce.up;
                if exist('ny','var')&&~isempty(ny)
                Ycell{folds,1}=Ytot{folds,1}(:,ny);  
                else
                Ycell{folds,1}=Ytot{folds,1};
                end
%                         if  obj.trunc_negative==1
%                           Y{folds,1}(Y{folds,1}<0)=0;
%                         end
        
                Y=Ycell;


end






end
function  [He_p]=Hermite(p_order)
%    H(0,X) = 1,
%    H(1,X) = X,
%    H(N,X) = X * H(N-1,X) - (N-1) * H(N-2,X)

        % polynomial
        He_p    = cell(p_order,1);
        He_p{1} = 1;       % H_1 = 1
        He_p{2} = [1 0];   % H_2 = x
        for n = 2:p_order
           He_p{n+1} = [He_p{n} 0] - (n-1)*[0 0 He_p{n-1}];   % recursive formula
        end
end
function Ypsi=Psi_solve(pce,x)
        
        xcell=cell(pce.p_order+1,pce.M);
        for i=1:pce.M
            for j=1:pce.p_order+1
        xcell{j,i}=x(:,i);
            end
        end

        Ypsicell=cellfun(@(x,y)polyval(x,y),pce.Psi,xcell,'UniformOutput',0);
        Ypsi=ones(size(x,1),pce.P);
        for i=1:pce.P
           for j=1:pce.M
              Ypsi(:,i)=Ypsi(:,i).*Ypsicell{pce.alpha(i,j)+1,j};
           end
        end
end
% -----------------------------------------------------------------------------
% Below you will find Vectorized Finite Element functions from the article:
% Talal Rahman, Jan Valdman, Fast MATLAB assembly of FEM matrices in 2D and
% 3D: nodal elements. Applied Mathematics and Computation 219, 7151-7158 (2013) 
% download: https://www.mathworks.com/matlabcentral/fileexchange/27826
% please cite the article when using the codes

function [M]=mass_matrixP1_2D(elements, areas,coeffs)
% Copyright (c) 2013, Talal Rahman, Jan Valdman
%coeffs can be only P0 (elementwise constant) function 
%represented by a collumn vector with size(elements,1) entries
%if coeffs is not provided then coeffs=1 is assumed globally
%Note: P1 coeffs needs a higher integration rule (not implemented yet)

Xscalar=kron(ones(1,3),elements); Yscalar=kron(elements,ones(1,3)); 

if (nargin<4)
    Zmassmatrix=kron(areas,reshape((ones(3)+eye(3))/12,1,9)); 
else
    if numel(coeffs)==size(elements,1) %P0 coefficients
        Zmassmatrix=kron(areas.*coeffs,reshape((ones(3)+eye(3))/12,1,9)); 
    else %P1 coefficients
        M1=[6 2 2; 2 2 1; 2 1 2]/60;
        M2=M1([3,1,2],[3,1,2]);
        M3=M2([3,1,2],[3,1,2]);
            
        Zmassmatrix=kron(areas.*coeffs(elements(:,1)),reshape(M1,1,9)) ...
                   +kron(areas.*coeffs(elements(:,2)),reshape(M2,1,9)) ...
                   +kron(areas.*coeffs(elements(:,3)),reshape(M3,1,9));

    end
        
end

M=sparse(Xscalar,Yscalar,Zmassmatrix);
end

function [A,areas]=stiffness_matrixP1_2D(elements,coordinates,coeffs)
% Copyright (c) 2013, Talal Rahman, Jan Valdman
%coeffs can be either P0 (elementwise constant) or P1 (elementwise nodal) function 
%represented by a collumn vector with size(elements,1) or size(coordinates,1) entries
%if coeffs is not provided then coeffs=1 is assumed globally
NE=size(elements,1); %number of elements
DIM=size(coordinates,2); %problem dimension
%particular part for a given element in a given dimension
NLB=3; %number of local basic functions, it must be known!
coord=zeros(DIM,NLB,NE);
for d=1:DIM
    for i=1:NLB
        coord(d,i,:)=coordinates(elements(:,i),d);
    end
end   
IP=[1/3;1/3];
[dphi,jac] = phider(coord,IP,'P1'); 
dphi = squeeze(dphi); 
areas=abs(squeeze(jac))/factorial(DIM);
if (nargin<3)
    Z=astam(areas',amtam(dphi,dphi));  
else
    if numel(coeffs)==size(coordinates,1)  %P1->P0 averaging
        coeffs=evaluate_average_point(elements,coeffs);
    end  
    Z=astam((areas.*coeffs)',amtam( dphi,dphi));
end
Y=reshape(repmat(elements,1,NLB)',NLB,NLB,NE);
%copy this part for a creation of a new element
X=permute(Y,[2 1 3]);
A=sparse(X(:),Y(:),Z(:));  
end

function amb = amsm (ama,smx)
% Copyright (c) 2013, Talal Rahman, Jan Valdman
% ama: ama(1:nx,1:ny,1:nz)
% smx: smx(1:ny,1:nk)
% amb: amb(1:nx,1:nk,1:nz)
[nx,ny,nz] = size(ama);
[ny,nk]    = size(smx);
amb     = zeros(nx,nk,nz);
for col = 1:nk
    amb(:,col,:) = amsv(ama,smx(:,col));
end
end

function avb = amsv (ama,svx)
% Copyright (c) 2013, Talal Rahman, Jan Valdman
[nx,ny,nz] = size(ama);
avx = svx(:).';
avx = avx(ones(nx,1),:,ones(nz,1));
avb = ama .* avx;
avb = sum(avb,2);
end

function amb = amtam (amx,ama)
% Copyright (c) 2013, Talal Rahman, Jan Valdman
% ama: ama(1:nx,1:ny,1:nz)
% amx: amx(1:nx,1:nk,1:nz)
% amb: amb(1:nk,1:ny,1:nz)
[nx,ny,nz] = size(ama);
[nx,nk,nz] = size(amx);
amb     = zeros(nk,ny,nz);
for row = 1:nk
    amb(row,:,:) = avtam(amx(:,row,:),ama);
end
end

function amb = astam (asx,ama)
% Copyright (c) 2015, Talal Rahman, Jan Valdman
[nx,ny,nz] = size(ama);
asx = reshape(asx,1,1,nz);
asx = asx(ones(nx,1),ones(ny,1),:);
amb = ama .* asx;
end

function avb = avtam (avx,ama)
% Copyright (c) 2013, Talal Rahman, Jan Valdman
[nx,ny,nz] = size(ama);
avx = avx(:,ones(ny,1),:);
avb = ama .* avx;
avb = sum(avb,1);
end

function amb = smamt (smx,ama)
% Copyright (c) 2013, Talal Rahman, Jan Valdman
% ama: ama(1:nx,1:ny,1:nz)
% amx: amx(1:nx,1:nk,1:nz)
% amb: amb(1:nk,1:ny,1:nz)
[ny,nx,nz] = size(ama);
[nk,nx]    = size(smx);
amb     = zeros(nk,ny,nz);
for row = 1:nk
    amb(row,:,:) = svamt(smx(row,:),ama);
end
end

function avb = svamt (svx,ama)
% Copyright (c) 2013, Talal Rahman, Jan Valdman
% ama: ama(1:ny,1:nx,1:nz)
% svx: svx(1,1:nx)
% avb: avb(1,1:ny,1:nz)
[ny,nx,nz] = size(ama);
avx = svx;
avx = avx(ones(ny,1),:,ones(nz,1));
avb = ama .* avx;
avb = sum(avb,2);
avb = reshape(avb,1,ny,nz);
end

function [dphi,detj,jac] = phider (coord,point,etype)
% Copyright (c) 2013, Talal Rahman, Jan Valdman
% PHIDER Returns the gradients of the basis functions
%        with respect to the local coordinates (x,y,...).
% coord : coord(nod,nos,noe), the local coordinates of the
%         nodes which the shape functions are associated with.
% point : point(nod,nop), the coordinates of the
%         points on the reference element.
%  dphi : dphi(nod,nos,nop,noe), the gradients of the
%         shape functions (second) at all points (third)
%         with respect to the local cordinates.
%   jac : jac(nod,nod,nop,noe), the Jacobian matrices
%         at all nop points.
%  detj : detj(1,nop,noe), determinants of the Jacobian matrices
%         at all nop points
% etype : 'P0','P1','P2', etc., the element type.
%         Note:
%         nod - dimension of the element.
%         nop - number of points.
%         nos - number of shape functions.
%         noe - number of elements.

jacout = 'no';
if nargout >2, jacout = 'yes'; end
detout = 'no';
if nargout >1, detout = 'yes'; end
nod = size(coord,1);
nop = size(point,2);
nos = size(coord,2);
noe = size(coord,3);
% Derivatives with respect to the reference
% coordinates (xi,eta,...).
dshape = shapeder(point,etype);
if strcmp(jacout, 'yes'), jac  = zeros(nod,nod,nop,noe); end
if strcmp(detout,'yes'),  detj = zeros(1,nop,noe); end
dphi = zeros(nod,nos,nop,noe);
for poi = 1:nop   
    tjac              = smamt(dshape(:,:,poi),coord);
    [tjacinv,tjacdet] = aminv(tjac);
    dphi(:,:,poi,:)   = amsm(tjacinv,dshape(:,:,poi));
    if strcmp(jacout, 'yes')
       jac(:,:,poi,:) = tjac;
    end
    if strcmp(detout, 'yes')
       detj(1,poi,:) = abs(tjacdet);
    end
end
end

function  elements2midpoint=evaluate_average_point(elements,coordinates,which_elements_number)
        
dummy=coordinates(elements(:,1),:);
for i=2:size(elements,2)
    dummy=dummy+coordinates(elements(:,i),:);
end
elements2midpoint=dummy/size(elements,2);

if nargin==3
   elements2midpoint=elements2midpoint(which_elements_number,:);
end

   
end


% -----------------------------------------------------------------------------
% the codes below were simplified by Jan Valdman in 2017 
% to work with P1 elements in 2D only

function [amb,dem] = aminv (ama)
% Copyright (c) 2013, Talal Rahman, Jan Valdman
% Modification: 2017 by Jan Valdman - only 2D case left
% ama: ama(1:nx,1:nx,1:nz)
% amb: amb(1:nx,1:nx,1:nz)
[nx,nx,nz] = size(ama);
if nx==2
    % Matrix elements.
    a = squeeze(ama(1,1,:)); b = squeeze(ama(1,2,:));
    c = squeeze(ama(2,1,:)); d = squeeze(ama(2,2,:));
    % Matrix determinant.
    dem = a.*d - b.*c;
    % Matrix inverse.
    amb = zeros(nx,nx,nz);
    amb(1,1,:) = d./dem;
    amb(2,2,:) = a./dem;
    amb(1,2,:) = - b./dem;
    amb(2,1,:) = - c./dem;
    dem = dem(:)';
else
    msg = 'Array operation for inverting matrices of dimension 2 is implemented.  ';
    msg = [msg 'Extend the code!'];
    error(msg); 
end
end

function [dshape] = shapeder (point,etype)
% Copyright (c) 2013, Talal Rahman, Jan Valdman
% Modification: 2017 by Jan Valdman - only P1 elements in 2D left
% SHAPEDER Returns the gradients of the shape functions with
%          respect to the reference coordinates (xi,eta,...).
%
%  point : point(nod,nop), the coordinates of the
%          points on the reference element.
% dshape : dshape(nod,nos,nop), the gradients of the
%          shape functions (second) at all points (third)
%          with respect to the reference cordinates.
%  etype : 'P0','P1','P2', etc., the element type.
%         
%          Note: 
%          nod - dimension of the element.
%          nop - number of points.
%          nos - number of shape functions.
nod = size(point,1);
nop = size(point,2);
switch nod
    case {2},   % 2-D elements
        l1 = point(1,:);
        l2 = point(2,:);
        l3 = 1 - l1 - l2;
        switch etype   
            case {'P1'},
            % Linear shape functions.
              dshape = [1 0 -1; 0 1 -1];
              dshape = reshape(dshape,6,1);
              dshape = dshape*ones(1,nop);
              dshape = reshape(dshape,2,3,nop);
            otherwise, error('Only P1 elements implemented.');
        end    
    otherwise, error('Only 2D implemented.');     
end
end