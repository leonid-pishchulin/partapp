% This file is part of the implementation of the human pose estimation model as described in the paper:
    
% Leonid Pishchulin, Micha Andriluka, Peter Gehler and Bernt Schiele
% Strong Appearance and Expressive Spatial Models for Human Pose Estimation
% IEEE International Conference on Computer Vision (ICCV'13), Sydney, Australia, December 2013

% Please cite the paper if you are using this code in your work.

% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without permission from the authors.  


% function vis_conf(log_dir, varargin)
%
% visualize part configuration
%
% 'log_dir' - experiment directory (without 'class' suffix)
%
% 'figidx' - matlab figure, where configuration will be show
% 
%
function vis_conf_mix(log_dir, varargin)

  [rootidx, figidx, ntypes, files] = process_options(varargin, 'rootidx', -1, 'figidx', 1, 'ntypes', 1, 'files', []);

  if (rootidx == -1) 
    RPIDX = load([log_dir '/params_mat/rootpart_idx.mat']);
    rootidx = RPIDX.rootpart_idx + 1;
  end    

  assert(rootidx >= 0);
  
  if (isempty(files))
    figure(figidx);
    clf;
  end
  
  class_dir = [log_dir, '/class'];
  if exist(class_dir, 'dir') == 0
    class_dir = log_dir;
  end
  
%   assert(exist(class_dir, 'dir') > 0);

  %class_dir = log_dir;
  if isempty(files)
      for tidx = 0:ntypes-1
          files = dir([class_dir '/joint_*_tidx_' num2str(tidx) '.mat']);
          subplot(1, ntypes, tidx+1);
          vis_conf_flip(class_dir, files, rootidx, false);
      end
  else
      
    vis_conf_flip(log_dir, files, rootidx, false);
%     vis_conf_flip(class_dir, files, rootidx, false);
  end
  %subplot(1, 2, 2);
  %vis_conf_flip(class_dir, files, rootidx, true);

function vis_conf_flip(class_dir, files, rootidx, flip)
  joint_files = cell(0);
  J = cell(0);

  for idx = 1:length(files)
    if length(files(idx).name) > 5 && strcmp(files(idx).name(1:5), 'joint') == 1
      joint_files{end+1} = [class_dir '/' files(idx).name];
      fprintf('%s\n', joint_files{end});
      J{end+1} = load(joint_files{end});
    end
  end

  njoints = length(J);
  nparts = njoints + 1;

  if rootidx == 0
    rootidx = nparts;
  end

  fprintf('found %d joints, rootidx: %d\n', length(joint_files), rootidx);
  if (length(joint_files) == 0)
    return;
  end

  T = eye(2);
  if flip
    T(1,1) = -1;
  end

  if (J{1}.type == 1) 
    for jidx = 1:njoints
      assert(J{jidx}.type == 1);
      offset_p = T*J{jidx}.offset_p;
      C = T*J{jidx}.C*T';

      %plot([0, offset_p(1)], [0, offset_p(2)], 'b-', 'LineWidth', 1);
      hold on;

      plot([0, offset_p(1)], [0, offset_p(2)], 'b.', 'MarkerSize', 20);

      drawellipse_kma(inv(C), offset_p(1), offset_p(2), 'r');
      hold on;
    end

  elseif (J{1}.type == 2)

    for jidx = 1:njoints
      J{jidx}.offset_p = T*J{jidx}.offset_p;
      J{jidx}.offset_c = T*J{jidx}.offset_c;
      J{jidx}.C = T*J{jidx}.C*T';

      if flip
        J{jidx}.rot_mean = -J{jidx}.rot_mean;
      end

    end

    fprintf('nparts: %d\n', nparts);

    part_pos = Inf(nparts, 2);
    part_rot = Inf(nparts, 1);

    part_pos(rootidx, :) = [0, 0];
    part_rot(rootidx, :) = 0;

    plotted_joints = false(njoints, 1);

    while ~all(plotted_joints)
      plotted_nothing = true;

      for jidx = 1:njoints
        assert(J{jidx}.type == 2);

        if (plotted_joints(jidx))
          continue;
        end

        parent_idx = J{jidx}.parent_idx;
        child_idx = J{jidx}.child_idx;

        if all(~isinf(part_pos(parent_idx, :)))
          fprintf('joint %d, parent_idx: %d, child_idx: %d, rot_mean: %f (%f), rot_sigma: %f (%f)\n', ...
                  jidx, parent_idx, child_idx, ...
                  J{jidx}.rot_mean, J{jidx}.rot_mean*180.0/pi, ...
                  J{jidx}.rot_sigma, J{jidx}.rot_sigma*180.0/pi);

          parent_pos = part_pos(parent_idx, :);

          parent_rot = part_rot(parent_idx);
          % compute joint position in world cs

          TP = [cos(parent_rot), -sin(parent_rot); ...
                sin(parent_rot), cos(parent_rot)];

%          if (flip) 
%            TP = T*TP;
%          end

          joint_pos = parent_pos + (TP*J{jidx}.offset_p)';

          % compute child position in world cs
          child_rot = parent_rot + J{jidx}.rot_mean;
          TC = [cos(child_rot), -sin(child_rot); ...
                sin(child_rot), cos(child_rot)];

%          if (flip)
%            TC = T*TC;
%          end

          child_pos = joint_pos - (TC*J{jidx}.offset_c)';

          % save child position and plot the joint

          part_pos(child_idx, :) = child_pos;
          part_rot(child_idx) = child_rot;

          plot([parent_pos(1), joint_pos(1), child_pos(1)], [parent_pos(2), joint_pos(2), child_pos(2)], 'b-', 'LineWidth', 3);%3
          hold on;
          %plot([parent_pos(1), child_pos(1)], [parent_pos(2), child_pos(2)], 'b.', 'LineWidth', 1, 'MarkerSize', 10);
          plot([parent_pos(1), child_pos(1)], [parent_pos(2), child_pos(2)], 'b.', 'MarkerSize', 30);%30

          r = 10;
          cov_mul_factor = 1;

          jointC = J{jidx}.C;

%          if (flip)
%            jointC = T*jointC*T';
%          end

          drawellipse_kma(inv(cov_mul_factor*jointC), joint_pos(1), joint_pos(2), 'r');
          hold on;
          
          child_rot_min = child_rot - J{jidx}.rot_sigma + pi/2;
          child_rot_max = child_rot + J{jidx}.rot_sigma + pi/2;

          vmin = r*[cos(child_rot_min), sin(child_rot_min)];
          vmax = r*[cos(child_rot_max), sin(child_rot_max)];

%          if (flip) 
%            vmin = T*vmin';
%            vmax = T*vmax';
%          end

          plot([joint_pos(1), joint_pos(1) + vmin(1)], [joint_pos(2), joint_pos(2) + vmin(2)], 'k-', 'LineWidth', 1);
          plot([joint_pos(1), joint_pos(1) + vmax(1)], [joint_pos(2), joint_pos(2) + vmax(2)], 'k-', 'LineWidth', 1);


          plotted_joints(jidx) = true;
          plotted_nothing = false;
        end

      end

      assert(plotted_nothing == false);
      
    end
    
    
  else
    fprintf('unknown joint type');
    assert(false);
  end

  hold on;
  plot(0, 0, 'k+');

  % tud pedestrians
  %axis([-75, 75, -105, 105]);

  % tud pedestrians (rot)
  axis([-75, 75, -70, 140]);
  %axis([-75, 75, -100, 140]);

  % buffy
  %axis([-75, 75, -100, 140]);

  % ramanan
  %axis([-50, 50, -60, 110]);

  axis ij;
  axis equal;
  hold off;


