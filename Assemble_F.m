function [model]=Assemble_F(model)
% Assemble RHS in 2D Elliptic PDE

if isa(model.f, 'function_handle')
    coord=sparse(2,3,model.mesh.N_e);
    for d=1:2
        for i=1:3
            coord(d,i,:)=model.mesh.nodes(model.mesh.elements(:,i),d);
        end
    end
    qpt = 1/3*sum(coord,2);
    g=model.f(squeeze(qpt(1,1,:)),squeeze(qpt(2,1,:)));
    F=model.mesh.areas/3.*g;
    model.F=sparse(reshape(model.mesh.elements(:,1:3),3*size(model.mesh.elements(:,1:3),1),1),1,reshape(repmat(F,1,3),3*size(model.mesh.elements(:,1:3),1),1),size(model.mesh.nodes,1),1);
else
    
    model.F=sparse(model.M*model.f);
    
end

end
