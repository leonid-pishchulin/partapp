function [det, all, det_no_nms, all_no_nms] = process_nms(image, model, thresh, bbox, overlap, chunksize)

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
  catch
    warning('no bounding box predictor found');
  end
  [det all] = clipboxes(image, det, all);
  det_no_nms = det;
  all_no_nms = all;
  I = nms(det, 0.5);
  det = det(I,:);
  all = all(I,:);
end
