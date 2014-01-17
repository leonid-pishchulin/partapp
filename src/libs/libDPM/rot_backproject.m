%
% imgwidth, imgheight - dimensions of the original image
% 2*padn, 2*padn - dimensions of the rotated image
% rod_deg - rotation degrees
% x, y - position in the rotated image
%

function [x, y] = rot_backproject(imgwidth_orig, imgheight_orig, imgwidth_rot, imgheight_rot, rot_deg, x, y)

  R = rot_matrix(rot_deg);
  
  v = [x; y] - [imgwidth_rot; imgheight_rot]/2;
  v = round(R*v + 0.5*[imgwidth_orig; imgheight_orig]);
  
  x = v(1);
  y = v(2);
  

function R = rot_matrix(rot_deg)

  rot_rad = (rot_deg / 180) * pi;
  c = cos(rot_rad);
  s = sin(rot_rad);

  R = [c, -s; s, c];