sz = size(ptCloud.Location);
pt = zeros(sz);
for i = 1:sz(1)
    if max(ptCloud.Location(i,:)) < 1
        pt(i,:) = ptCloud.Location(i,:);
    end   
end