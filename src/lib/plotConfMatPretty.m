function plotConfMatPretty(C,labels,titleStr,savePath)
% C        – confusion matrix   (integer counts)
% labels   – cellstr with class names (rows = cols = numel(labels))
% titleStr – string for figure title
% savePath – filename (.png or .svg) to write; empty = no save

cm = confusionchart(C , labels , ...
        'Normalization' ,'row-normalized', ...
        'RowSummary'   ,'row-normalized', ...
        'ColumnSummary','column-normalized');
cm.CellLabelFormat = '%.1f%%';
cm.Title           = titleStr;
drawnow
if ~isempty(savePath)
    exportgraphics(cm,savePath,'Resolution',300);
end
end
