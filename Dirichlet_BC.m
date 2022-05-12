function [model]=Dirichlet_BC(model)
% Apply Dirichlet BC in 2D Elliptic PDE example

b=model.mesh.boundary(model.mesh.boundary(:,3)<5,1:2);
b=unique(b);
model.Ab=model.A;
model.Fb=model.F;
model.Mb=model.M;

model.Mb(b,:)=0; model.Mb(:,b)=0; model.Ab(b,:)=0; model.Ab(:,b)=0; model.Fb(b)=0; % put zeros in boundary rows/columns of K and F
model.Ab(b,b)=speye(length(b),length(b)); % put I into boundary submatrix of K
model.Mb(b,b)=speye(length(b),length(b)); % put I into boundary submatrix of M


end