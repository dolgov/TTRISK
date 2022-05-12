function []=plot_nodalfield(model, field,tit)
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
