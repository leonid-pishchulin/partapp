% This file is part of the implementation of the human pose estimation model as described in the paper:
    
% L. Pishchulin, M. Andriluka, P. Gehler and B. Schiele
% Strong Appearance and Expressive Spatial Models for Human Pose Estimation
% IEEE Conference on Computer Vision and Pattern Recognition (ICCV'13), Sydney, Australia, December 2013

% Please cite the paper if you are using this code in your work.

% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without permission from the authors.  

function test_scoregrid_gui(test_annolist, log_dir, first_img, last_img)

  show_part_prob = false;
  show_log_part_prob = true;
  anneal_factor = 1.0;
  lp_ver = 0;


  % slider limits
  total_images = last_img - first_img + 1;

  %
  % pedestrians
  %
%  total_parts = 6;
%  total_scales = 11;
%  total_rotations = 6;
%  total_orientations = 2;
%  min_rot = -45;
%  max_rot = 45;
%  num_rot = 6;

  %
  % buffy
  %
%  total_parts = 6;
%  total_scales = 2;
%  total_rotations = 24;
%  total_orientations = 2;
%  min_rot = -180;
%  max_rot = 180;
%  num_rot = 24;

  %
  % PARSE
  %
  total_parts = 22;
  total_scales = 2;
  total_rotations = 48;
  total_orientations = 2;
  min_rot = -180;
  max_rot = 180;
  num_rot = 48;

  % layout parameters
  slider_x = 30;
  slider_y = 20;
  slider_width = 130;
  slider_height = 15;

  slider_step = 15;
  slider_vstep = 15;

  text_width = slider_width;
  text_height = slider_height;

  % initial values
  cur_imgidx = first_img;
  cur_pidx = 1;
  cur_scaleidx = 1;
  cur_rotidx = 1;
  cur_flip = 0;

  % create window
  fh = figure;
  num_sliders = 5;
  pos = get(fh, 'Position');
  pos(3) = 2*slider_x + num_sliders*(slider_width + slider_step);
  set(fh, 'Position', pos);

  % create gui controls 
  gc_slider1 = uicontrol(fh,'Style','slider',...
                         'Max', total_images, 'Min', 1, 'Value', cur_imgidx, ...
                         'SliderStep', [1.0/(total_images - 1) 1.0/(total_images - 1)], ...
                         'Position',[slider_x, slider_y, slider_width, slider_height], ...
                         'Callback', @slider1_callback);

  gc_text1 = uicontrol(fh,'Style','text',...
                       'String','Image: ',...
                       'Position',[slider_x, slider_y + slider_height + slider_vstep, text_width, text_height]);


  x2 = slider_x + slider_width + slider_step;
  gc_slider2 = uicontrol(fh,'Style','slider',...
                         'Max', total_parts, 'Min', 1, 'Value', cur_pidx, ...
                         'SliderStep', [1.0/(total_parts - 1) 1.0/(total_parts - 1)], ...
                         'Position',[x2, slider_y, slider_width, slider_height], ...
                         'Callback', @slider2_callback);

  gc_text2 = uicontrol(fh,'Style','text',...
                       'String','Part: ',...
                       'Position',[x2, slider_y + slider_height + slider_vstep, text_width, text_height]);

  x3 = slider_x + 2*(slider_width + slider_step);
  gc_slider3 = uicontrol(fh,'Style','slider',...
                         'Max', total_scales, 'Min', 1, 'Value', cur_scaleidx, ...
                         'SliderStep', [1.0/(total_scales - 1) 1.0/(total_scales - 1)], ...
                         'Position',[x3, slider_y, slider_width, slider_height], ...
                         'Callback', @slider3_callback);


  gc_text3 = uicontrol(fh,'Style','text',...
                       'String','Scale: ',...
                       'Position',[x3, slider_y + slider_height + slider_vstep, text_width, text_height]);


  x4 = slider_x + 3*(slider_width + slider_step);
  gc_slider4 = uicontrol(fh,'Style','slider',...
                         'Max', total_rotations, 'Min', 1, 'Value', cur_rotidx, ...
                         'SliderStep', [1.0/(total_rotations - 1) 1.0/(total_rotations - 1)], ...
                         'Position',[x4, slider_y, slider_width, slider_height], ...
                         'Callback', @slider4_callback);


  gc_text4 = uicontrol(fh,'Style','text',...
                       'String','Rotation: ',...
                       'Position',[x4, slider_y + slider_height + slider_vstep, text_width, text_height]);

  x5 = slider_x + 4*(slider_width + slider_step);
  gc_slider5 = uicontrol(fh,'Style','slider',...
                         'Max', total_orientations, 'Min', 1, 'Value', cur_rotidx, ...
                         'SliderStep', [1.0/(total_orientations - 1) 1.0/(total_orientations - 1)], ...
                         'Position',[x5, slider_y, slider_width, slider_height], ...
                         'Callback', @slider5_callback);


  gc_text5 = uicontrol(fh,'Style','text',...
                       'String','Orientation: ',...
                       'Position',[x5, slider_y + slider_height + slider_vstep, text_width, text_height]);


  % positions are in the format: distance from left, distance from bottom, width, height
  % all relative to window size

  rel_axis_height = 0.65;
  if ~show_part_prob
    ah1 = axes('Parent',fh,'Position',[.05 .25 .4 rel_axis_height]);
    ah2 = axes('Parent',fh,'Position',[.55 .25 .4 rel_axis_height]);
  else
    ah1 = axes('Parent',fh,'Position',[.05 .25 .25 rel_axis_height]);
    ah2 = axes('Parent',fh,'Position',[.35 .25 .30 rel_axis_height]);
    ah3 = axes('Parent',fh,'Position',[.70 .25 .30 rel_axis_height]);
  end

  img = [];
  DATA = cell(0);
  PART_PROB_DATA = cell(0);

  on_switch_image();

  function str_imgname = get_image_filename(imgidx)
      assert(length(test_annolist) >= imgidx);

      str_imgname = test_annolist(imgidx).image.name;
  end

  function str_gridname = get_scoregrid_filename(imgidx, pidx, flip)    
    str_gridname = [log_dir '/test_scoregrid/imgidx' num2str(imgidx-1) ...
                    '-pidx' num2str(pidx - 1) ...
                    '-o' num2str(flip) ...
                    '-scoregrid.mat'];
  end


  function str_gridname = get_probgrid_filename(imgidx, pidx, flip)    
    str_gridname = [log_dir '/test_partprob/imgidx' num2str(first_img + imgidx - 1) ...
                    '-pidx' num2str(pidx - 1) ...
                    '-o' num2str(flip)];
    if lp_ver ~= 0
      str_gridname = [str_gridname '-ver' num2str(lp_ver)];
    end

    str_gridname = [str_gridname '-logP.mat'];
  end  

  function on_switch_image()
    set(gc_text1, 'String', ['Image: ' num2str(cur_imgidx-1)]);

    img_filename = get_image_filename(cur_imgidx);
    fprintf('image: %s\n', img_filename);
    img = imread(img_filename);

    axes(ah1);
    imagesc(img);
    colormap gray;
    freezeColors(ah1);

    for pidx = 1:total_parts
      part_scoregrid_name = get_scoregrid_filename(cur_imgidx, pidx, cur_flip);
      fprintf('loading "%s" ...\n', part_scoregrid_name);
      DATA{pidx} = load(part_scoregrid_name);

      if show_part_prob
        part_probgrid_name = get_probgrid_filename(cur_imgidx, pidx, cur_flip);

        if exist(part_probgrid_name) 
          fprintf('loading "%s" ...\n', part_probgrid_name);
          PART_PROB_DATA{pidx} = load(part_probgrid_name);
        else
          fprintf('missing "%s" ...\n', part_probgrid_name);
          PART_PROB_DATA{pidx} = [];
        end

      end
    end

    % update text control 
    on_switch_part();

  end

  function on_switch_part()
    part_names = {'lleg1', ...
                  'lleg2', ...
                  'torso'};

    set(gc_text2, 'String', ['Part: ' num2str(cur_pidx-1)]);

    on_switch_rotation_scale();
  end

  function on_switch_rotation_scale()
    assert(length(DATA) == total_parts);

    axes(ah2);
    grid2 = DATA{cur_pidx}.cell_scoregrid{cur_scaleidx, cur_rotidx};
    imagesc(grid2, [-0.8,0.3]);
    fprintf('max score: %f\n', max(max(DATA{cur_pidx}.cell_scoregrid{cur_scaleidx, cur_rotidx})));

    colorbar;
    colormap jet;
    freezeColors(ah2);

    if show_part_prob
      axes(ah3);
      if ~isempty(PART_PROB_DATA{cur_pidx})
        logP = PART_PROB_DATA{cur_pidx}.img_logP{cur_scaleidx, cur_rotidx};
        if show_log_part_prob
          fprintf('showing unnormalized log probilities\n');
          P = logP;
        else
          assert(false);
          fprintf('showing probilities separately normalized for each scale and orientation\n');
          fprintf('anneal_factor %f\n', anneal_factor);
          P = exp(anneal_factor*(logP - min(min(logP))));
          fprintf('max(max(P)) before normalization: %.7f\n', max(max(P)));
          P = P/sum(sum(P));
        end

        fprintf('max(max(P)): %.7f\n', max(max(P)));
        imagesc(P);
        colorbar;
      else
        fprintf('warning: could not load log probabilities\n');
      end

    end

    set(gc_text3, 'String', ['Scale: ' num2str(cur_scaleidx)]);
    set(gc_text4, 'String', ...
    ['Rotation: ' num2str(value_from_index(cur_rotidx - 1, num_rot, min_rot, max_rot)) ' (' num2str(cur_rotidx) ')']);
  end

  function slider1_callback(hObject, eventdata)
    cur_imgidx = round(get(hObject, 'Value'));
    fprintf('current image: %d, orientation: %d\n', cur_imgidx, cur_flip);
    on_switch_image();    
  end

  function slider2_callback(hObject, eventdata)
    fprintf('slider2_callback\n');
    cur_pidx = round(get(hObject, 'Value'))
    on_switch_part();
  end 

  function slider3_callback(hObject, eventdata)
    fprintf('slider3_callback\n');
    cur_scaleidx = round(get(hObject, 'Value'))
    on_switch_rotation_scale();
  end

  function slider4_callback(hObject, eventdata)
    fprintf('slider4_callback\n');
    cur_rotidx = round(get(hObject, 'Value'))
    on_switch_rotation_scale();
  end

  function slider5_callback(hObject, eventdata)
    fprintf('slider5_callback\n');
    cur_flip = round(get(hObject, 'Value')) - 1
    
    on_switch_image();
  end

end


function value = value_from_index(idx, num_steps, minval, maxval)

  assert(idx >= 0 && idx < num_steps);

  step_size = (maxval - minval)/num_steps;
  value = minval + step_size*(0.5 + idx);

  minval
  idx
  value
end