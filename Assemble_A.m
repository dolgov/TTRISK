function [model]=Assemble_A(model,randfield)
% Assemble stiffness matrix in the 2D Elliptic PDE example

coord=zeros(2,3,model.mesh.N_e);
for d=1:2
    for i=1:3
        coord(d,i,:)=model.mesh.nodes(model.mesh.elements(:,i),d);
    end
end   
qpt = 1/3*sum(coord,2);
model.mesh.qpt=squeeze(qpt)';

if nargin<2
% evaluate coefficient using funciton a 
coeff=model.a(model.mesh.qpt(:,1),model.mesh.qpt(:,2));
else
coeff=randfield;  
end

[model.A, model.mesh.areas]=stiffness_matrixP1_2D(model.mesh.elements(:,1:3),model.mesh.nodes(:,1:2),coeff);

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


