function [track_res,output] = L1TrackingBPR_APGup(s_frames, paraT)
 

%% Initialize templates T
%-Generate T from single image
init_pos = paraT.init_pos;
n_sample=paraT.n_sample;
sz_T=paraT.sz_T;
rel_std_afnv = paraT.rel_std_afnv;
nT=paraT.nT;

%generate the initial templates for the 1st frame
img = imread(s_frames{1});
if(size(img,3) == 3)
    img = rgb2gray(img);
end
[T,T_norm,T_mean,T_std] = InitTemplates(sz_T,nT,img,init_pos);
norms = T_norm.*T_std; %template norms
occlusionNf = 0;

%% L1 function settings
angle_threshold = paraT.angle_threshold;
para.Lambda = paraT.lambda;
para.nT = paraT.nT;
para.Lip = paraT.Lip;
para.Maxit = paraT.Maxit;

dim_T	= size(T,1);	%number of elements in one template, sz_T(1)*sz_T(2)=12x15 = 180
A		= [T eye(dim_T)]; %data matrix is composed of T, positive trivial T.
alpha = 50;%this parameter is used in the calculation of the likelihood of particle filter
aff_obj = corners2affine(init_pos, sz_T); %get affine transformation parameters from the corner points in the first frame
map_aff = aff_obj.afnv;
aff_samples = ones(n_sample,1)*map_aff;

T_id	= -(1:nT);	% template IDs, for debugging
fixT = T(:,1)/nT; % first template is used as a fixed template

%Temaplate Matrix
Temp = [A fixT];
Dict = Temp'*Temp;
Temp1 = [T,fixT]*pinv([T,fixT]);

%% Tracking

% initialization
nframes	= length(s_frames);
track_res	= zeros(6,nframes);
Time_record = zeros(nframes,1);
Coeff = zeros(size([A fixT],2),nframes);
Min_Err = zeros(nframes,1);
count = zeros(nframes,1);
Time = zeros(n_sample,1); % L1 norm time recorder
ratio = zeros(nframes,1);% energy ratio

for t = 1:nframes
%     fprintf('Frame number: %d/%d \n',t, nframes);
    
    img_color	= imread(s_frames{t});
    if(size(img_color,3) == 3)
        img     = double(rgb2gray(img_color));
    else
        img     = double(img_color);
    end
    
    tic
    %-Draw transformation samples from a Gaussian distribution
    map_aff = local_sanitize_afnv(map_aff);
    aff_samples = local_sanitize_samples(aff_samples, map_aff);
    sc			= sqrt(sum(map_aff(1:4).^2)/2);
    std_aff		= rel_std_afnv.*[1, sc, sc, 1, sc, sc];
    map_aff		= map_aff + 1e-14;
    aff_samples = draw_sample(aff_samples, std_aff); %draw transformation samples from a Gaussian distribution
    aff_samples = local_sanitize_samples(aff_samples, map_aff);
    
    %-Crop candidate targets "Y" according to the transformation samples
    [Y, Y_inrange] = crop_candidates(im2double(img), aff_samples(:,1:6), sz_T);
    if(sum(Y_inrange==0) == n_sample)
%         sprintf('Target is out of the frame!\n');
    end
    
    [Y,Y_crop_mean,Y_crop_std] = whitening(Y);	 % zero-mean-unit-variance
    [Y, Y_crop_norm] = normalizeTemplates(Y); %norm one
    
    %-L1-LS for each candidate target
    eta_max	= -inf;
    q   = zeros(n_sample,1); % minimal error bound initialization
    id_max = 1;
    c_max = zeros(size([A fixT],2), 1);
   
    % first stage L2-norm bounding    
    for j = 1:n_sample
        if Y_inrange(j)==0 || sum(abs(Y(:,j)))==0
            continue;
        end
        
        % L2 norm bounding
        q(j) = norm(Y(:,j)-Temp1*Y(:,j));
        q(j) = exp(-alpha*q(j)^2);
    end
    q(~isfinite(q)) = 0;
    %  sort samples according to descend order of q
    [q,indq] = sort(q,'descend');    
    
    % second stage
    p	= zeros(n_sample,1); % observation likelihood initialization
    n = 1;
    tau = 0;
    while (n<n_sample)&&(q(n)>=tau)        

        [c] = APGLASSOup(Temp'*Y(:,indq(n)),Dict,para);
        
        D_s = (Y(:,indq(n)) - [A(:,1:nT) fixT]*[c(1:nT); c(end)]).^2;%reconstruction error
        p(indq(n)) = exp(-alpha*(sum(D_s))); % probability w.r.t samples
        tau = tau + p(indq(n))/(2*n_sample-1);%update the threshold
        
        if(sum(c(1:nT))<0) %remove the inverse intensity patterns
            continue;
        elseif(p(indq(n))>eta_max)
            id_max	= indq(n);
            c_max	= c;
            eta_max = p(indq(n));
            Min_Err(t) = sum(D_s);
        end
        n = n+1;
    end
    p(~isfinite(p)) = 0;
    
    count(t) = n;    
    
    % resample according to probability
    if ~isfinite(eta_max)
        valid_q = find(q > 0, 1);
        if isempty(valid_q)
            id_max = 1;
        else
            id_max = indq(valid_q);
        end
        c_max = zeros(size([A fixT],2), 1);
    end
    map_aff = aff_samples(id_max,1:6); %target transformation parameters with the maximum probability
    map_aff = local_sanitize_afnv(map_aff);
    a_max	= c_max(1:nT);
    [aff_samples, ~] = resample(aff_samples,p,map_aff); %resample the samples wrt. the probability
    aff_samples = local_sanitize_samples(aff_samples, map_aff);
    [~, indA] = max(a_max);
    min_angle = images_angle(Y(:,id_max),A(:,indA));
    ratio(t) = norm(c_max(nT:end-1));
    Coeff (:,t) = c_max;    
    
     %-Template update
     occlusionNf = occlusionNf-1;
     level = 0.03;
    if( min_angle > angle_threshold && occlusionNf<0 )        
%         disp('Update!')
        trivial_coef = c_max(nT+1:end-1);
        trivial_coef = reshape(trivial_coef, sz_T);
        
        trivial_coef = im2bw(trivial_coef, level);

        se = [0 0 0 0 0;
            0 0 1 0 0;
            0 1 1 1 0;
            0 0 1 0 0'
            0 0 0 0 0];
        trivial_coef = imclose(trivial_coef, se);
        
        cc = bwconncomp(trivial_coef);
        stats = regionprops(cc, 'Area');
        areas = [stats.Area];
        
        % occlusion detection 
        if (max(areas) < round(0.25*prod(sz_T)))        
            % find the tempalte to be replaced
            [~,indW] = min(a_max(1:nT));
        
            % insert new template
            T(:,indW)	= Y(:,id_max);
            T_mean(indW)= Y_crop_mean(id_max);
            T_id(indW)	= t; %track the replaced template for debugging
            norms(indW) = Y_crop_std(id_max)*Y_crop_norm(id_max);
        
            [T, ~] = normalizeTemplates(T);
            A(:,1:nT)	= T;
        
            %Temaplate Matrix
            Temp = [A fixT];
            Dict = Temp'*Temp;
            Temp1 = [T,fixT]*pinv([T,fixT]);
        else
            occlusionNf = 5;
            % update L2 regularized term
            para.Lambda(3) = 0;
        end
    elseif occlusionNf<0
        para.Lambda(3) = paraT.lambda(3);
    end
    
    Time_record(t) = toc;

    %-Store tracking result
    track_res(:,t) = map_aff';
    
end
 
output.time = Time_record; % cpu time of APG method for each frame
output.minerr = Min_Err; % reconstruction error for each frame
output.coeff = Coeff;  % best coefficients for each frame
output.count = count;  % particles used to calculate the L1 minimization in each frame
output.ratio = ratio;  % the energy of trivial templates
end

function afnv = local_sanitize_afnv(afnv)
if all(isfinite(afnv))
    afnv(1) = max(afnv(1), eps);
    afnv(4) = max(afnv(4), eps);
    return;
end

afnv = [1, 0, 0, 1, 0, 0];
end

function samples = local_sanitize_samples(samples, fallback_afnv)
invalid_rows = ~all(isfinite(samples), 2);
if any(invalid_rows)
    samples(invalid_rows, :) = ones(sum(invalid_rows), 1) * fallback_afnv;
end
samples(:, 1) = max(samples(:, 1), eps);
samples(:, 4) = max(samples(:, 4), eps);
end
