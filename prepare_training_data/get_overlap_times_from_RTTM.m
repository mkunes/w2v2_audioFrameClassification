function overlaps_times = get_overlap_times_from_RTTM(rttm_filename,minNonOLLen,minOLLen,minPauseLen_sameSpk)
% get speech overlaps from RTTM, as a Nx2 array of overlap start times and end times
%
% ---
%
% Changelog:
%   2023-03-21
%     - now compatible with Octave v6.4.0 (not tested in any other versions)
%     - added epsilon 1e-10 to some floating point comparisons to ensure Matlab and Octave give the exact same results
%       (They seem to handle floating point arithmetic slightly differently)
%
% Marie Kunesova (https://github.com/mkunes)
% 2022

if nargin < 4
    minPauseLen_sameSpk = 0;
end


% workaround for imperfect floating point arithmetic
epsilon = 1e-10; 
if minPauseLen_sameSpk > epsilon
    minPauseLen_sameSpk = minPauseLen_sameSpk - epsilon;
end
if minOLLen > epsilon
    minOLLen = minOLLen - epsilon;
end
minNonOLLen = minNonOLLen + epsilon;


% read the entire RTTM
fileID = fopen(rttm_filename, 'r');
labels = textscan(fileID,'%s %s %d %f %f %s %s %s %[^\n\r]');
    % RTTM file entries are formatted like this:
    %   "SPEAKER" audio_ID channel_num start_time duration "<NA>" "<NA>" speaker_id [one or two more "<NA>"s]
    % e.g.:
    %   SPEAKER	2412-153948-0008_777-126732-0053	1	0.33	4.01	<NA>	<NA>	777	<NA>
fclose(fileID);

if isempty(labels{1}{end}) % happens in Octave 6.4.0, but not in Matlab
    for ii = 1:numel(labels)
        labels{ii} = labels{ii}(1:end-1);
    end
end

nLabels = numel(labels{1});
speakers = unique(labels{8});
nSpk = numel(speakers);
utterances = zeros(nLabels,3);
for iLabel = 1:nLabels
    if ~strcmp(labels{1}{iLabel},'SPEAKER')
        continue;        
    end
    if labels{5}(iLabel) <= 0
        % AMI has at least 2 files with a negative segment duration (EN2005a, ES2011c) - ignore those segments
        warning('Negative or zero utterance duration in file %s at time %g (duration %g). Ignoring this entry.',...
            rttm_filename,labels{4}(iLabel),labels{5}(iLabel));
        continue;
    end
    spkName = labels{8}{iLabel};
    [~,spkId] = ismember(spkName,speakers);
    utterances(iLabel,1) = spkId;
    utterances(iLabel,2) = labels{4}(iLabel); % start time
    utterances(iLabel,3) = labels{4}(iLabel) + labels{5}(iLabel); % end time = start time + duration
end

% sort the utterances by start time, just in case
[~,order] = sort(utterances(:,2));
utterances = utterances(order,:);

% mark down all speaker changes (start of utterance, end of utterance) and who speaks
changes = zeros(2 * nLabels, 2);
nChanges = 0;
for iSpk = 1:nSpk
    utterances_iSpk = utterances(utterances(:,1) == iSpk,:);
    
    nChanges = nChanges + 1;
    changes(nChanges,1) = utterances_iSpk(1,2); % start time of the speaker's first utterance
    changes(nChanges,2) = iSpk; % the speaker's ID, indicating START
    
    for iSeg = 2:size(utterances_iSpk,1)
        if utterances_iSpk(iSeg,2) - utterances_iSpk(iSeg-1,3) >= minPauseLen_sameSpk % pause between the same speaker's utterances
            % the pause is long enough -> include it as two speaker changes
            %   (otherwise it gets ignored)
            nChanges = nChanges + 2;
            changes(nChanges-1,1) = utterances_iSpk(iSeg-1,3); % end time of the speaker's previous utterance
            changes(nChanges-1,2) = -iSpk; % -1 * speaker's ID, indicating END
            changes(nChanges,1) = utterances_iSpk(iSeg,2); % start time of the speaker's current utterance
            changes(nChanges,2) = iSpk; % the speaker's ID, indicating START
        end
    end
    
    nChanges = nChanges + 1;
    changes(nChanges,1) = utterances_iSpk(end,3); % end time of the speaker's last utterance
    changes(nChanges,2) = -iSpk; % -1 * speaker's ID, indicating END
end

[~,order] = sort(changes(1:nChanges,1));
changes = changes(order,:);

% generate a list of overlaps by tracking the number of active speakers after each speaker change
overlaps_times = zeros(size(changes));
active_speakers = [];
nOverlaps = 0;
isOverlap = false;

for iChng = 1:size(changes,1)
    spkID = changes(iChng,2);
    t = changes(iChng,1);
    if spkID > 0 % start of utterance
        
        if ismember(spkID,active_speakers)
            % Note: some RTTM files can genuinely have "overlaps" with the same speaker. 
            %   But! minPauseLen_sameSpk means that such cases will be handled before we reach this point
            %   -> if this error triggers, something's really wrong
            error('Speaker %d was already active at time %d!',spkID,t);
        else
            % add spkID to active speakers
            active_speakers(end+1) = spkID; %#ok<AGROW>
            if numel(active_speakers) > 1 && ~isOverlap
                % overlap starts here
                isOverlap = true;
                
                if nOverlaps == 0 || t - overlaps_times(nOverlaps,2) >= minNonOLLen
                    % if the time since last overlap is long enough, add this as a new overlap
                    %   (otherwise they get merged)
                    
                    nOverlaps = nOverlaps + 1;
                    overlaps_times(nOverlaps,1) = changes(iChng,1);
                end
            end
        end
    end
    if spkID < 0 % end of utterance
        spkID = -spkID;
        [~,loc] = ismember(spkID,active_speakers);
        
        if loc == 0
            % this also shouldn't ever happen
            error('Speaker %d was not active at time %g!',spkID,changes(iChng,1));
        else
            % remove spkID from active speakers
            active_speakers = [active_speakers(1:loc-1) active_speakers(loc+1:end)];
            if numel(active_speakers) < 2 && isOverlap
                % overlap ends here
                isOverlap = false;
                overlaps_times(nOverlaps,2) = changes(iChng,1);
            end
        end
    end
end
if isOverlap
    overlaps_times(nOverlaps,2) = Inf;
end
overlaps_times = overlaps_times(1:nOverlaps,:);


durations = overlaps_times(:,2) - overlaps_times(:,1); % duration of each overlap
overlaps_times = overlaps_times(durations >= minOLLen,:); % only keep overlaps that are long enough

end

