function [model]=Solve_FEM(model)
% Solve linear equation in 2D Elliptic example
model.y=model.Ab\model.Fb;
end
