function [dets, boxes, info] = imgdetect(input, model, thresh, bbox, overlap, chunksize, delta_scales)

% Wrapper that computes detections in the input image.
%
% input    input image
% model    object model
% thresh   detection score threshold
% bbox     ground truth bounding box
% overlap  overlap requirement
% chunksize number of threads % introduced by Leonid
if nargin < 4
  bbox = [];
end

if nargin < 5
  overlap = 0;
end

if nargin < 6
  chunksize = 4;
end

if nargin < 7
  delta_scales = -1;
end

% we assume color images
input = color(input);

% get the feature pyramid
pyra = featpyramid(input, model);
% changed by Leonid
if (delta_scales > -1)
    min_root = max(1, model.interval+1-delta_scales);
    max_root = min(length(pyra.scales), model.interval+1+delta_scales);
    min_parts = max(1, 2*model.interval+1-delta_scales);
    max_parts = min(length(pyra.scales), 2*model.interval+1+delta_scales);
    root_levels = min_root:max_root;
    parts_levels = min_parts:max_parts;
    pyra.valid_levels = unique(sort([root_levels parts_levels]));
else
    pyra.valid_levels = 1:length(pyra.scales);
end
%DEBUG
% pyra.feat = pyra.feat([9:13 19:23]);
% pyra.scales = pyra.scales([9:13 19:23]);
% model.interval = 2;
%DEBUG

[dets, boxes, info] = gdetect(pyra, model, thresh, bbox, overlap, chunksize);
