% annotations - either path to annoation file (*.al) or structure array with annotations
%
% output_dir - directory where rescaled annotations will be saved
%
% varargin: method - method used by imresize
%

function [annotations, output_annolist] = rescale_data(annotations, pathToExp, outputdir, scale, varargin)

fprintf('rescale_data()\n');

method = process_options(varargin, 'method', 'bicubic');
annolist_filename = [];

if ischar(annotations)
    
    annolist_filename = annotations;
    assert(exist(annolist_filename) == 2);
    
    fprintf('loading %s\n', annolist_filename);
    annotations = loadannotations(annolist_filename);
    
else
    assert(isstruct(annotations));
end

fprintf('rescaling images\n');
for imgnum = 1:length(annotations)
    fprintf('.');
    [path, imgname, ext] = fileparts(annotations(imgnum).image.name);
    
    img = imread([pathToExp '/' annotations(imgnum).image.name]);
    
    img = imresize(img, scale, method);
    
    resimgname = [outputdir '/' imgname ext];
    
    imwrite(img, resimgname);
    
    annotations(imgnum).image.name = resimgname;
    for ar = 1:length(annotations(imgnum).annorect)
        annotations(imgnum).annorect(ar).x1 = round(scale*annotations(imgnum).annorect(ar).x1);
        annotations(imgnum).annorect(ar).y1 = round(scale*annotations(imgnum).annorect(ar).y1);
        annotations(imgnum).annorect(ar).x2 = round(scale*annotations(imgnum).annorect(ar).x2);
        annotations(imgnum).annorect(ar).y2 = round(scale*annotations(imgnum).annorect(ar).y2);
        
        if isfield(annotations(imgnum).annorect(ar), 'annopoints')
            
            if length(annotations(imgnum).annorect(ar).annopoints) == 1 && isfield(annotations(imgnum).annorect(ar).annopoints, 'point')
                for j = 1:length(annotations(imgnum).annorect(ar).annopoints.point)
                    annotations(imgnum).annorect(ar).annopoints.point(j).x = ...
                        round(scale * annotations(imgnum).annorect(ar).annopoints.point(j).x);
                    annotations(imgnum).annorect(ar).annopoints.point(j).y = ...
                        round(scale * annotations(imgnum).annorect(ar).annopoints.point(j).y);
                end
            end
            
            if isfield(annotations(imgnum).annorect(ar), 'objpos')
                annotations(imgnum).annorect(ar).objpos.x = round(scale * annotations(imgnum).annorect(ar).objpos.x);
                annotations(imgnum).annorect(ar).objpos.y = round(scale * annotations(imgnum).annorect(ar).objpos.y);
            end
            
        end
        
    end
    if (~mod(imgnum, 100))
        fprintf(' %d/%d\n',imgnum,length(annotations));
    end
end
fprintf('\ndone\n');

% save annotationlist
[path, name, ext] = splitpathext(annolist_filename);
output_annolist = [outputdir '/' name '-' num2str(scale) 'x.al'];
fprintf('saving %s\n', output_annolist);
saveannotations(annotations, output_annolist, 1.0, 1.0, true);

end