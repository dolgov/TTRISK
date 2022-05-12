function [Sol]=cvar_localsolve(i, XAX1, Ai, XAX2, XY1, yi, XY2, tol, sol_prev, nu)
% Solve reduced KKT systems in block AMEn

mBMB{1} = XAX1{2,2}{1}; mBMB{2} = Ai{2,2}{1}; mBMB{3} = XAX2{2,2}{1};
r0 = size(mBMB{1},1);
ny = size(mBMB{2},2);
r1 = size(mBMB{3},2);

A{1} = XAX1{2,1}{1}; A{2} = Ai{2,1}{1}; A{3} = XAX2{2,1}{1};
% Trace prec
tic;
Ap = reshape(A{3}, [], r1^2);
ind = (1:r1)+r1*(0:r1-1);
Ap = mean(Ap(:, ind), 2);
Ap = sparse(full_matrix_2d(A{1},A{2},Ap)); % size r0*ny
toc;

AT{1} = XAX1{1,2}{1}; AT{2} = Ai{1,2}{1}; AT{3} = XAX2{1,2}{1};

Hyy{1} = XAX1{1,1}{1}; Hyy{2} = Ai{1,1}{1}; Hyy{3} = XAX2{1,1}{1};
% Trace prec
Hyyp = reshape(Hyy{3}, [], r1^2);
Hyyp = mean(Hyyp(:, ind), 2);
Hyyp = sparse(full_matrix_2d(Hyy{1},Hyy{2},Hyyp));
HyyLR = {zeros(r0,r0,1); sparse(ny, ny); zeros(1,r1,r1)};
HyyLRp = sparse(r0*ny, r0*ny);
if (numel(Ai{1,1})==2)
    % Hyy is given in two terms, the second one is low-rank correction in
    % space
    HyyLR{1} = XAX1{1,1}{2}; HyyLR{2} = Ai{1,1}{2}; HyyLR{3} = XAX2{1,1}{2};
    HyyLRp = reshape(HyyLR{3}, [], r1^2);
    HyyLRp = mean(HyyLRp(:, ind), 2);
    if (i==1)
        HyyLRp = reshape(HyyLR{2}.Rfactor.', [], size(HyyLRp,1)) * sparse(HyyLRp);
        HyyLRp = reshape(HyyLRp, [], ny);
        HyyLRp = lrmatrix(HyyLR{2}.Lfactor, HyyLRp.');
    else
        HyyLRp = sparse(full_matrix_2d(HyyLR{1},HyyLR{2},HyyLRp));
    end
end

mBMBp = reshape(mBMB{3}, [], r1^2);
ind = (1:r1)+r1*(0:r1-1);
mBMBp = mean(mBMBp(:, ind), 2);
mBMBp = mBMB{2}*mBMBp; % just a sparse or rank-1 matrix
% For i==1 this mBMBp is a sparse matrix ready to use
if (i>1)
    % The left block might also be low-rank
    [U,S,V] = svd(mBMB{1}, 'econ');
    S = diag(S);
    rmbm = my_chop2(S, S(1)*tol);
    mBMBp.Lfactor = kron(mBMBp.Lfactor, U(:,1:rmbm)*diag(S(1:rmbm)));
    mBMBp.Rfactor = kron(mBMBp.Rfactor, V(:,1:rmbm));
end


% Remaining matrices
Hty{1} = XAX1{1,3}{1}; Hty{2} = Ai{1,3}{1}; Hty{3} = XAX2{1,3}{1};
Htyw{1} = XAX1{3,1}{1}; Htyw{2} = Ai{3,1}{1}; Htyw{3} = XAX2{3,1}{1};

tic;
if (issparse(Hty{2}))
    R = size(Hty{1},3);
    Hty{2} = reshape(Hty{2}, R*ny^2, []);
    Hty{2} = Hty{2}(1:R*ny, :);
else
    Hty{2} = Hty{2}(:,:,1,:);
end
Htyp = reshape(Hty{3}, [], r1^2);
Htyp = mean(Htyp(:, ind), 2);
Htyp = full_matrix_2d(Hty{1}, Hty{2}, Htyp);

if (issparse(Htyw{2}))
    R = size(Htyw{1},3);
    Htyw{2} = reshape(Htyw{2}, R*ny, []);
    Htyw{2} = Htyw{2}(1:R, :);
else
    Htyw{2} = Htyw{2}(:,1,:,:);
end
Htywp = reshape(Htyw{3}, [], r1^2);
Htywp = mean(Htywp(:, ind), 2);
Htywp = full_matrix_2d(Htyw{1}, Htyw{2}, Htywp);

Htt = XAX1{3,3}{1}(1,1,1)*Ai{3,3}{1}(1)*XAX2{3,3}{1}(1,1,1);
Http = Htt*speye(r0);
Htt = {eye(r0), Htt, reshape(eye(r1),1,r1,r1)};
toc;


% RHS
fp = XY1{1}{1}*reshape(yi{1}{1},size(XY1{1}{1},2),[]);
fp = reshape(fp, r0*ny, [])*XY2{1}{1};
fp = fp(:);

fy = XY1{2}{1}*reshape(yi{2}{1},size(XY1{2}{1},2),[]);
fy = reshape(fy, r0*ny, [])*XY2{2}{1};
fy = fy(:);

ft = XY1{3}{1}*reshape(yi{3}{1}(:,1,:), size(XY1{3}{1},2), [])*XY2{3}{1};
ft = ft(:);

% Initial guess
y = sol_prev(1:r0*ny*r1);
p = sol_prev(1*r0*ny*r1+1:2*r0*ny*r1);
t = reshape(sol_prev(2*r0*ny*r1+1:3*r0*ny*r1), r0, ny, r1);
t = t(:,1,:);
t = t(:);


rhs = [fp;fy;ft];

pos = cumsum([1;numel(y);numel(p);numel(t)]);


% Preconditioning parts
tic;
% Syy = Hyyp + HyyLRp - (Htyp/Http)*Htywp;
% renorm = sqrt(norm(mBMBp+0,1)/norm(Syy,1));
renorm = sqrt( norm2est(@(x)mBMBp*x, r0*ny) / norm2est(@(x)Hyyp*x + HyyLRp*x - Htyp*(Http\(Htywp*x)), r0*ny) );

SleftP = Ap' + renorm*Hyyp;
if (i>1)
    SleftP = SleftP + renorm*HyyLRp; % -renorm*(Hty/Htt)*Htyw;
end
SrightP = Ap;
if (i==1)
    SrightP = Ap - mBMBp/renorm;
end
% Left vectors in SMW
if (i>1)
    SleftCorr = SleftP\(-renorm*Htyp/Http);
    SleftCorr = SleftCorr/(eye(size(Htyp,2)) + Htywp*SleftCorr);    
else
    SleftCorr = SleftP\(renorm*[HyyLRp.Lfactor -Htyp]/blkdiag(Http, eye(size(HyyLRp.Lfactor,2))));
    Htywp = [HyyLRp.Rfactor.'; Htywp];
    SleftCorr = SleftCorr/(eye(size(Htyp,2)+size(HyyLRp.Lfactor,2)) + Htywp*SleftCorr);
end
if (i==1)
    SrightCorr = [];
else
    SrightCorr = SrightP\(-mBMBp.Lfactor/renorm);
    SrightCorr = SrightCorr/(eye(size(mBMBp.Lfactor,2)) + mBMBp.Rfactor'*SrightCorr); % size r0*r1
end
toc;

Sol_red = fgmres(@(X,t)kktmv(Hyy,HyyLR,AT,Hty,mBMB,A,Htyw,Htt,pos,X), rhs, tol, 'P', @(F,t)precvec(F,pos,r1,Ap,mBMBp,SleftP,SleftCorr,SrightP,SrightCorr,Htyp,Htt{2},Htywp), 'x0', [y;p;t], 'restart', 1024, 'max_iters', 1, 'verb', 1);

Sol = zeros(r0,ny,r1, 3);
Sol(:,:,:,1) = reshape(Sol_red(pos(1):pos(2)-1), r0, ny, r1);
Sol(:,:,:,2) = reshape(Sol_red(pos(2):pos(3)-1), r0, ny, r1);
Sol(:,1,:,3) = reshape(Sol_red(pos(3):pos(4)-1), r0, 1, r1);
Sol = Sol(:);
end


function [f]=kktmv(Hyy,HyyLR,AT,Hty,mBMB,A,Htyw,Htt,pos,x)
y = x(pos(1):pos(2)-1);
p = x(pos(2):pos(3)-1);
t = x(pos(3):pos(4)-1);

fp = kron3mult(Hyy,y)+kron3mult(HyyLR,y) + kron3mult(AT,p) + kron3mult(Hty,t);
fy = kron3mult(A,y) + kron3mult(mBMB,p);
ft = kron3mult(Htyw,y) + kron3mult(Htt,t);

f = [fp;fy;ft];
end


function [B] = full_matrix_2d(A1,A2,A3)
% A3 should have mode size 1
[n1,m1,r1] = size(A1);
[r2,~,~] = size(A3);
if (issparse(A2))
    [r1n2, m2r2] = size(A2);
    n2 = round(r1n2 / r1);
    m2 = round(m2r2 / r2);
else
    [~,n2,m2,~] = size(A2);
end

A2 = reshape(A2, [], r2);
A2 = A2*sparse(A3);
A2 = reshape(A2, r1, []);

B = sparse(n1*n2, m1*m2);
for j=1:r1
    B = B  +  kron(reshape(A2(j,:), n2, m2),   sparse(A1(:,:,j)));
end
end

function [y]=kron3mult(A,x)
[~,m1,r1] = size(A{1});
[r2,~,m3] = size(A{3});
if (issparse(A{2}))
    m2 = size(A{2},2)/r2;
else
    [~,~,m2,~] = size(A{2});
end
y = reshape(x, [], m3).'; % rem m1*m2
y = reshape(A{3}, [], m3) * y;
y = reshape(y, r2, []); % rem n3*m1*m2
y = y.';
y = reshape(y, [], m2*r2); % rem n3*m1
y = y.';
y = reshape(A{2}, [], m2*r2) * y;
y = reshape(y, [], m1); % rem r1*n2*n3
y = y.';
y = reshape(y, m1*r1, []); % rem n2*n3
y = reshape(A{1}, [], m1*r1) * y;
y = y(:);
end


function [Sol]=precvec(Rhs,pos,r1,A,mBMBp,Sleft,SleftCorr,Sright,SrightCorr,Hty,Htt,Htyw)
fy = Rhs(pos(1):pos(2)-1);
fy = reshape(fy, [], r1);
fp = Rhs(pos(2):pos(3)-1);
fp = reshape(fp, [], r1);
ft = Rhs(pos(3):pos(4)-1);
ft = reshape(ft, [], r1);

t = Htt\ft;

p = fp - Hty*t;
% Sleft carries also a low-rank -renorm*(Hty/Htt)*Htyw
% Solve with SMW
p = Sleft\p;
p = p - SleftCorr*(Htyw*p);

p = A*p;

% Sright carries also (mB/Mu)*mBw/renorm
p = Sright\p;
if (~isempty(SrightCorr))
    p = p - SrightCorr*(mBMBp.Rfactor'*p); 
end

y = fy - mBMBp*p;
y = A\y;

y = y(:);
p = p(:);
t = t(:);

Sol = [y;p;t];
end


% Simple Arnoldi method to estimate the 2-norm
function [L] = norm2est(A,n)
V = randn(n,16);
V(:,1) = V(:,1)/norm(V(:,1));
Lprev = 0;
H = zeros(size(V,2)+1, size(V,2));
for i=2:min(size(V,2)+1,n+1)
    W = A(V(:,i-1));
    for j=1:i-1
        H(j,i-1) = V(:,j)'*W;
        W = W - V(:,j)*H(j,i-1);
    end
    H(i,i-1) = norm(W);
    L = norm(H(1:i, 1:i-1));
    if (abs(L/Lprev-1)<1e-6) || (i==(size(V,2)+1))
        break;
    end
    Lprev = L;
    V(:,i) = W/H(i,i-1);    
end
end
