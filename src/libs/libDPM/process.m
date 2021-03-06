function [det, all] = process(image, model, thresh, bbox, overlap, chunksize, delta_scales)

% bbox = process(image, model, thresh)
% Detect objects that score above a threshold, return bonding boxes.
% If the threshold is not included we use the one in the model.
% This should lead to high-recall but low precision.

globals;

if nargin < 3
  thresh = model.thresh
end

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

[det, all] = imgdetect(image, model, thresh, bbox, overlap, chunksize, delta_scales);

if ~isempty(det)
  try
    % attempt to use bounding box prediction, if available
    bboxpred = model.bboxpred;
    [det all] = clipboxes(image, det, all);
    [det all] = bboxpred_get(bboxpred, det, reduceboxes(model, all));
    
%     [dets, boxes] = imgdetect(im, model, model.thresh);
%     if ~isempty(boxes)
%       boxes = reduceboxes(model, boxes);
%       [dets boxes] = clipboxes(im, dets, boxes);
%       I = nms(dets, thresh);
%       boxes1{i} = dets(I,[1:4 end]);
%       parts1{i} = boxes(I,:);
%     else
%       boxes1{i} = [];
%       parts1{i} = [];
%     end
    
  catch
    warning('no bounding box predictor found');
  end
  [det all] = clipboxes(image, det, all);
  I = nms(det, 0.5);
  det = det(I,:);
  all = all(I,:);
end
