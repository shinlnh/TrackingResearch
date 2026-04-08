seq={
'vot15_bag'
'vot15_ball1'
};

for s=1:numel(seq)
   runTracker(seq{s},1);
end
