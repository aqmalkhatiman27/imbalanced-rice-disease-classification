function [learnableLayer,classLayer] = findLayersToReplace(lgraph)
% Returns the last learnable layer (FC or Conv) and the final
% classification layer inside a layerGraph.

if ~isa(lgraph,'nnet.cnn.LayerGraph')
    error('Input must be a layerGraph');
end

learnableLayer = [];
classLayer     = [];

% walk backwards to find the last FC or multi-filter Conv layer
for ii = numel(lgraph.Layers):-1:1
    L = lgraph.Layers(ii);
    if isa(L,'nnet.cnn.layer.FullyConnectedLayer') || ...
       (isa(L,'nnet.cnn.layer.Convolution2DLayer') && L.NumFilters > 1)
        learnableLayer = L;
        break
    end
end
if isempty(learnableLayer)
    error('No learnable (FC / Conv) layer found.');
end

% find the ClassificationOutputLayer
for ii = numel(lgraph.Layers):-1:1
    if isa(lgraph.Layers(ii),'nnet.cnn.layer.ClassificationOutputLayer')
        classLayer = lgraph.Layers(ii);
        break
    end
end
end
