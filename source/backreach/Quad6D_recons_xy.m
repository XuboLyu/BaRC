function [targetXY] = Quad6D_recons_xy(gMin, gMax, gN,targetX, targetY,tMax)
Xdim = 1;
Ydim = 3;

gX = createGrid(gMin(Xdim), gMax(Xdim), gN(Xdim));
gY = createGrid(gMin(Ydim), gMax(Ydim), gN(Ydim));

%disp(gX.vs{1});



vfs.gs = {};
vfs.gs{end+1} = gX;
vfs.gs{end+1} = gY;
vfs.tau = 0:0.05:tMax;
vfs.datas = {};
vfs.datas{end+1} = targetX;
vfs.datas{end+1} = targetY;
vfs.dims = {};
vfs.dims{end+1} = 1;
vfs.dims{end+1} = 2;


range_lower = [-10,-10];
range_upper = [10,10];

vf = reconSC(vfs,range_lower,range_upper,0);
targetXY = vf.data;

end