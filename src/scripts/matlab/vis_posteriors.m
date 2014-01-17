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

% function vis_posteriors(varargin)
%
% visualize part posteriors obtained with partapp
% 
% log_dir - directory where results of the experiment are stored (log_dir)
% imgidx - index of the image (this is 0-based index)
% figidx - matlab figure where results will be shown
%
function vis_posteriors(varargin)

  [imgidx, figidx, flip, scaleidx, log_dir, conv_part_mask] = process_options(varargin, ...
                                                    'imgidx', 0, ...
                                                    'figidx', 1, ...
                                                    'flip', false, ...
                                                    'scaleidx', 0, ...
                                                    'log_dir', [], ...
                                                    'conv_part_mask', true);

  assert(~isempty(log_dir));

  % load experiment/dataset dependent parameters (saved by partapp during initialization)
  % 
  % - number of parts
  % - part dimensions 
  % - discretization of part rotation (min, max and numsteps)
  %

  S = load([log_dir '/params_mat/num_parts.mat']);
  exp_params.num_parts = S.num_parts;

  S = load([log_dir '/params_mat/part_dims.mat']);
  exp_params.part_dims = S.part_dims;

  disp(exp_params.part_dims);

  S = load([log_dir '/params_mat/rotation_params.mat']);

  exp_params.min_part_rotation = S.min_part_rotation;
  exp_params.max_part_rotation = S.max_part_rotation;
  exp_params.num_rotation_steps = S.num_rotation_steps;

  if conv_part_mask
    exp_params.conv_part_mask = true;
    exp_params.gauss_sigma = -1.0  
    %exp_params.gauss_sigma = 2.0  
    exp_params.vis_multiplier = 2e3;

  else
    exp_params.conv_part_mask = false;
    exp_params.gauss_sigma = 2.0;

    %exp_params.gauss_sigma = -1.0;
    exp_params.vis_multiplier = 300;
  end

  disp(exp_params);

  grid_name_base = [log_dir '/part_marginals/log_part_posterior_final_imgidx'];
  grid_name_base = [grid_name_base num2str(imgidx) '_scaleidx' num2str(scaleidx) '_o' num2str(flip)];

  assert(length(scaleidx) == 1);
  all_parts_grid2 = load_log_prob_grid(grid_name_base, exp_params);

  vis_posteriors_helper(exp_params, all_parts_grid2, figidx);
  
function all_parts_grid2 = load_log_prob_grid(fname_base, exp_params)

  all_parts_grid2 = cell(0);

  for pidx = 1:exp_params.num_parts
      
      % DEBUG: vis only for 10 parts in 22 parts model
      if (~ismember(pidx, [2 4 7 9 11 12 14 16 19 21]))
         continue;
      end
      
    fname = [fname_base '_pidx' num2str(pidx-1) '.mat']
    
    LP = load(fname);
    grid2 = draw_part_rects(exp_params, LP.log_prob_grid, pidx);

    minval = min(min(grid2));
    maxval = max(max(grid2));
    fprintf('pidx %d, minval: %f (%f), maxval %f (%f)\n', pidx-1, minval, exp(minval), maxval, exp(maxval));

    grid2 = exp(grid2);

    all_parts_grid2{end+1} = grid2;
  end

function grid2 = draw_part_rects(exp_params, grid3, pidx)

  minval = min(min(min(grid3)));
  maxval = max(max(max(grid3)));

  assert(length(size(grid3)) == 3);

  num_rot = size(grid3, 1);
  fprintf('num_rotations: %d\n', num_rot);

  grid3 = exp(grid3);

  if exp_params.conv_part_mask
    for rotidx = 1:num_rot
      step_size = (exp_params.max_part_rotation - exp_params.min_part_rotation)/exp_params.num_rotation_steps;

      rotval = exp_params.min_part_rotation + step_size*(0.5 + rotidx - 1);
      
      part_width = round(0.5*exp_params.part_dims(pidx, 1));
      part_height = round(0.5*exp_params.part_dims(pidx, 2));

      T = ones(part_height, part_width);

      T = imrotate(T, -rotval);
      A = squeeze(grid3(rotidx, :, :));
      A = conv2(A, T, 'same');
      grid3(rotidx, :, :) = A;
    end
  end

  grid2 = sum(grid3, 1);
  grid2 = squeeze(grid2);

  if exp_params.gauss_sigma > 0
    grid2 = cv_gaussianfilter(grid2, exp_params.gauss_sigma);
  end

  grid2 = log(grid2);
  grid2(grid2 == -Inf) = -100;
