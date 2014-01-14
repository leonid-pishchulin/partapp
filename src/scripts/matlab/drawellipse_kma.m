%
% from http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/display_features.m
% 
function drawellipse_kma(Mi,i,j,col)
  hold on;
  [v e]=eig(Mi);

  l1=1/sqrt(e(1));

  l2=1/sqrt(e(4));

  alpha=atan2(v(4),v(3));
  s=1;
  t = 0:pi/50:2*pi;
  y=s*(l2*sin(t));
  x=s*(l1*cos(t));

  xbar=x*cos(alpha) + y*sin(alpha);
  ybar=y*cos(alpha) - x*sin(alpha);
  plot(ybar+i,xbar+j,col,'LineWidth',1);

  hold off;


