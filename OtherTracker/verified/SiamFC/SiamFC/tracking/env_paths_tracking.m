function p = env_paths_tracking(p)

    tracking_dir = fileparts(mfilename('fullpath'));
    p.net_base_path = [tracking_dir filesep];
%     p.seq_base_path = '/home/lab/Desktop/test/SiamFC/sequences/';
    p.seq_vot_base_path = '/path/to/VOT/evaluation/sequences/'; % (optional)
    p.stats_path = ''; % (optional)

end
