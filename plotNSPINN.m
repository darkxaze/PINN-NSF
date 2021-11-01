clear
close all

set(0,'defaulttextinterpreter','latex')
%load Cylinder3D.mat
load NS3D_beltrami.mat
load NS3D_beltrami_data.mat
fig = figure();
set(fig,'units','normalized','outerposition',[0 0 1 1])

for num = 100:100 %size(t_star,1)
    disp(num)
    
    clf
    
    subplot(2,2,1)
    plot_isosurface_griddata(x_star, y_star, z_star, U_star(:,num),'$x$','$y$','$z$','Regressed $u(t,x,y,z)$')
    drawnow()
          
    subplot(2,2,2)
    plot_isosurface_griddata(x_star, y_star, z_star, V_star(:,num),'$x$','$y$','$z$','Regressed $v(t,x,y,z)$')
    drawnow()
    
    subplot(2,2,3)
    plot_isosurface_griddata(x_star, y_star, z_star, W_star(:,num),'$x$','$y$','$z$','Regressed $w(t,x,y,z)$')
    drawnow()
    
    subplot(2,2,4)
    plot_isosurface_griddata(x_star, y_star, z_star, P_star(:,num),'$x$','$y$','$z$','Regressed $p(t,x,y,z)$')
    drawnow()
    
    %%%

end

% addpath ~/export_fig
% export_fig ./Cylinder_3D_results.png -r300