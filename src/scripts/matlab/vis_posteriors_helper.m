%
% convert posterior probabilities to images 
%  - different parts -> different hues
%  - higher probability -> higher intensity
% 
% add the images of different parts in RGB space 
% and show them
%

function vis_posteriors_helper(exp_params, all_parts_grid2, figidx)
  nparts = length(all_parts_grid2);

  img_width = size(all_parts_grid2{1}, 2);
  img_height = size(all_parts_grid2{1}, 1);

  sumimg = zeros(img_height, img_width, 3);

  for pidx = 1:nparts
    part_hue = pidx*(1/nparts);

    hsvgrid = get_hsv_grid(all_parts_grid2{pidx}, part_hue);
    rgbgrid{pidx} = hsv2rgb(hsvgrid);
    sumimg = sumimg + rgbgrid{pidx};
  end

  if figidx > 0
    figure(figidx)
    clf;
  end

  imshow(exp_params.vis_multiplier*sumimg);


function hsvgrid = get_hsv_grid(grid2, part_hue)

  part_saturation = 1.0;

  img_width = size(grid2, 2);
  img_height = size(grid2, 1);

  minval = min(min(grid2));
  maxval = max(max(grid2));
  Z = sum(sum(grid2));
  fprintf('minval: %f, maxval: %f, Z: %f\n', minval, maxval, Z);

  hsvgrid = zeros(img_height, img_width, 3);

  norm_grid = grid2 / Z;

  hsvgrid(:, :, 1) = part_hue;
  hsvgrid(:, :, 2) = part_saturation;
  hsvgrid(:, :, 3) = norm_grid;
