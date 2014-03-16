function annotations = prepare_data_file(imgDirName)
% creates an annotation file in the format compatible with partapp
% imgDirName    directory with images

pngDir = [imgDirName '/png'];
if (~exist(pngDir, 'dir'))
    mkdir(pngDir);
end

files = dir(imgDirName);
annotations = struct('image', {}, 'imgnum', {});
num = 0;

for i = 1:length(files)
    [p,n,e] = fileparts(files(i).name);
    if (strcmp(e, '.jpg') || strcmp(e, '.JPG') || strcmp(e, '.png'))
        fname = [imgDirName '/' files(i).name];
        img = imread(fname);
        imwrite(img, [pngDir '/' n '.png']);
        nexti = length(annotations) + 1;
        annotations(nexti).image.name = [pngDir '/' n '.png'];
        num = num + 1;
        annotations(nexti).imgnum = num;
        % add fake bounding box
        annotations(nexti).annorect.x1 = 1;
        annotations(nexti).annorect.y1 = 1;
        annotations(nexti).annorect.x2 = 2;
        annotations(nexti).annorect.y2 = 2;
    end
end

saveannotations(annotations, [pngDir '/annotations.al']);

end