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

% annotations - annotation list
% outputfilename
% rescale_factor - rescale all annorects by this factor
% score_factor - multiply all scores by this factor
% abs_path - if false all image filenames will be saved as relative
%
%function saveannotations(annotations, outputfilename, rescale_factor, score_factor, abs_path)
function saveannotations(annotations, outputfilename, rescale_factor, score_factor, abs_path, skip_empty)
  if nargin < 3
    rescale_factor = 1;
    score_factor = 1;
    abs_path = true;
  end

  if exist('skip_empty', 'var') == 0
    skip_empty = false;
  end

  if rescale_factor ~= 1
    assert(false, 'currently unsupported due introduction of rotated rectangles');
  end
  
  outputfilename = deblank(outputfilename);

  fprintf('saving annotations to %s\n', outputfilename);
  fid = fopen(outputfilename, 'w');

  if fid == -1 
    error('can not open output file');
  end

  if isfield(annotations, 'annorect')
    if rescale_factor ~= 1
      annotations = annotations_rescale(annotations, rescale_factor);
    end

    annotations = annotations_multscores(annotations, score_factor);
  end

  for ai = 1:length(annotations)
    [imgpath, filename] = splitpath(annotations(ai).image.name);
    if ~abs_path
      annotations(ai).image.name = filename;
    else
      if isfield(annotations(ai).image, 'path')  
        if isempty(imgpath) && length(annotations(ai).image.path) > 0
            annotations(ai).image.name = [annotations(ai).image.path '/' annotations(ai).image.name];
        end
        annotations(ai).image = rmfield(annotations(ai).image, 'path');
      end
    end

  end
  
  if strcmp(outputfilename(end-1:end), 'al')
    fprintf('using xml format\n');
    fprintf(fid, '%s', annotations2xml(annotations,  skip_empty));
  else
    fprintf('using idl format\n');
    fprintf(fid, '%s', annotations2idl(annotations));
  end
  fclose(fid);
end

function annotations = annotations_rescale(annotations, rescale_factor)
  for ai = 1:length(annotations)
    for ri = 1:length(annotations(ai).annorect)
      hx = (annotations(ai).annorect(ri).x1 + annotations(ai).annorect(ri).x2)/2;
      hy = (annotations(ai).annorect(ri).y1 + annotations(ai).annorect(ri).y2)/2;
      dx = abs(annotations(ai).annorect(ri).x2 - annotations(ai).annorect(ri).x1)/2;
      dy = abs(annotations(ai).annorect(ri).y2 - annotations(ai).annorect(ri).y1)/2;
      annotations(ai).annorect(ri).x1 = round(hx - rescale_factor*dx);
      annotations(ai).annorect(ri).y1 = round(hy - rescale_factor*dy);
      annotations(ai).annorect(ri).x2 = round(hx + rescale_factor*dx);
      annotations(ai).annorect(ri).y2 = round(hy + rescale_factor*dy);
    end
  end
end

function annotations = annotations_multscores(annotations, score_factor)
  for ai = 1:length(annotations)
    for ri = 1:length(annotations(ai).annorect)
      if ~isfield(annotations(ai).annorect(ri), 'score')
        annotations(ai).annorect(ri).score = -1;
      else
        annotations(ai).annorect(ri).score = score_factor * annotations(ai).annorect(ri).score;
      end
    end
  end
end

function res = annotations2xml(annotations, skip_empty)
  res = [];
  nl_char = sprintf('\n');

  for ai = 1:length(annotations)
    include_anno = true;
    
    if skip_empty && length(annotations(ai).annorect) == 0
        include_anno = false;
    end
    
    if include_anno
      res = [res '<annotation>' nl_char struct2xml(annotations(ai)) '</annotation>' nl_char];
    end
  end
  res = ['<annotationlist>' nl_char res '</annotationlist>'];
end

function res = annotations2idl(annotations)
  res = [];
  nl_char = sprintf('\n');

  for ai = 1:length(annotations)
    res = [res '"' annotations(ai).image.name '"'];
    if length(annotations(ai).annorect) > 0
      res = [res ':'];
      for ri = 1:length(annotations(ai).annorect)
        str = sprintf('(%d, %d, %d, %d):%d', ...
                      annotations(ai).annorect(ri).x1, ...
                      annotations(ai).annorect(ri).y1, ...
                      annotations(ai).annorect(ri).x2, ...
                      annotations(ai).annorect(ri).y2, ...
                      annotations(ai).annorect(ri).score);
                      
        res = [res str];

        if ri ~= length(annotations(ai).annorect)
          res = [res ', '];
        end
      end

      if ai == length(annotations)
        res = [res '.'];
      else
        res = [res ';'];
      end

    else
      res = [res ':(0, 0, 0, 0):-1;'];
    end

    res = [res nl_char];
  end

end

