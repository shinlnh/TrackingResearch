function feats = ptav_extract_features(img, rois, pixel_means, imageSz)
%PTAV_EXTRACT_FEATURES Extract verifier features via the Python OpenCV DNN backend.

global ptav_py_backend;
if isempty(ptav_py_backend)
    ptav_init_backend();
end

if isempty(rois) || size(rois, 1) < 1 || size(rois, 2) < 5
    feats = zeros(0, 0, 'double');
    return;
end

np = py.importlib.import_module('numpy');
py_img = np.array(uint8(img));
py_rois = np.array(double(rois));
py_means = np.array(double(pixel_means));
py_feats = ptav_py_backend.extract_features(py_img, py_rois, int32(imageSz), py_means);
feats = double(py_feats);
end
