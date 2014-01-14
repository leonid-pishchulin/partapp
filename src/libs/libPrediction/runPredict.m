idxPredMode = 1;
nClus = 12;
idxPred = -1;
saveTo = '/BS/leonid-pose/work/experiments-iccv13/log_dir/lsp-1000-0125-48-marginals-22-parts-mix-rot-torso-pos-prior-head-45x45-ic-rot-pos-pwise-12-comp-OC';
% poseletResponcesTrainDir = '/BS/leonid-pose/work/experiments-iccv13/log_dir/lsp-1000-poselets-200-min-clus-0005-strip-40-dpm-torso-pos-prior/resp_train';
% poseletResponcesTestDir = '/BS/leonid-pose/work/experiments-iccv13/log_dir/lsp-1000-poselets-200-min-clus-0005-strip-40-dpm-torso-pos-prior/resp_test';
poseletResponcesTrainDir = '/BS/leonid-people-3d/work/experiments-icps/log_dir/lsp-1000-poselets-200-min-clus-0005-strip-40-dpm-torso-pos-prior/resp_train';
poseletResponcesTestDir = '/BS/leonid-people-3d/work/experiments-icps/log_dir/lsp-1000-poselets-200-min-clus-0005-strip-40-dpm-torso-pos-prior/resp_test/';
torsoPosTrainDir = '/BS/leonid-pose/work/experiments-iccv13/log_dir/lsp-1000-0125-48-marginals-22-parts-torso-pos-prior-test-train-OC/part_marginals/';
torsoPosTestDir = '/BS/leonid-pose/work/experiments-iccv13/log_dir/lsp-1000-0125-48-marginals-22-parts-mix-rot-torso-pos-prior-head-45x45-OC/part_marginals/';
bUseOracle = 0;
nParts = 22;
torsoPartIdx = 10;
annolistTrain = '/BS/leonid-people-3d/work/data/lsp_dataset/images/png/h200/lsp-train-human-full-h200-OC-add-flipped.al';
annolistTest = '/BS/leonid-people-3d/work/data/lsp_dataset/images/png/h200/lsp_test_human_lr-rescaled_4viewpoints-OC.al';

predict_factors(idxPredMode,nClus,idxPred,saveTo,poseletResponcesTrainDir,poseletResponcesTestDir,torsoPosTrainDir,torsoPosTestDir,bUseOracle,nParts, ...
                torsoPartIdx, annolistTrain, annolistTest)
