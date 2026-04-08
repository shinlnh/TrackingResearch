function [r center] = drawAffine(afnv, tsize)

rect= round(aff2image(afnv', tsize));
inp	= reshape(rect,2,4);
x_coords = inp(2, :);
y_coords = inp(1, :);

left = min(x_coords);
right = max(x_coords);
top = min(y_coords);
bottom = max(y_coords);

center=[(left + right)/2, (top + bottom)/2];

r = [left, top, right-left+1, bottom-top+1];

% rect= round(aff2image(afnv', tsize));
% inp	= reshape(rect,2,4);
% 
% topleft_r = inp(1,1);
% topleft_c = inp(2,1);
% botleft_r = inp(1,2);
% botleft_c = inp(2,2);
% topright_r = inp(1,3);
% topright_c = inp(2,3);
% botright_r = inp(1,4);
% botright_c = inp(2,4);
% p = line([topleft_c, topright_c], [topleft_r, topright_r]);
% set(p, 'Color', color); set(p, 'LineWidth', linewidth); set(p, 'LineStyle', '-');
% p = line([topright_c, botright_c], [topright_r, botright_r]);
% set(p, 'Color', color); set(p, 'LineWidth', linewidth); set(p, 'LineStyle', '-');
% p = line([botright_c, botleft_c], [botright_r, botleft_r]);
% set(p, 'Color', color); set(p, 'LineWidth', linewidth); set(p, 'LineStyle', '-');
% p = line([botleft_c, topleft_c], [botleft_r, topleft_r]);
% set(p, 'Color', color); set(p, 'LineWidth', linewidth); set(p, 'LineStyle', '-');
