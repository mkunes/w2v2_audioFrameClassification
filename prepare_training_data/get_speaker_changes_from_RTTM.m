function changes = get_speaker_changes_from_RTTM(rttm_filename,minPauseLen)
% get speaker changes from RTTM, as a 1D array of times
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


% workaround for imperfect floating point arithmetic
epsilon = 1e-10; 
if minPauseLen > epsilon
    minPauseLen = minPauseLen - epsilon;
end

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
segments = zeros(nLabels,3);
for iLabel = 1:nLabels
    if ~strcmp(labels{1}{iLabel},'SPEAKER')
        continue;        
    end
    spkName = labels{8}{iLabel};
    [~,spkId] = ismember(spkName,speakers);
    segments(iLabel,1) = spkId;
    segments(iLabel,2) = labels{4}(iLabel); % start time
    segments(iLabel,3) = labels{4}(iLabel) + labels{5}(iLabel); % end time = start time + duration
end

% sort the utterances by start time, just in case
[~,order] = sort(segments(:,2));
segments = segments(order,:);

changes = zeros(2 * nLabels, 1);
nChanges = 0;
for iSpk = 1:nSpk
    segments_iSpk = segments(segments(:,1) == iSpk,:);
    
    nChanges = nChanges + 1;
    changes(nChanges) = segments_iSpk(1,2); % start time of the speaker's first utterance
    
    for iSeg = 2:size(segments_iSpk,1)
        if segments_iSpk(iSeg,2) - segments_iSpk(iSeg-1,3) >= minPauseLen - epsilon % pause between the same speaker's utterances
                                                                            
            
            % the pause is long enough -> include it as two speaker changes
            %   (otherwise it gets ignored)
            nChanges = nChanges + 2;
            changes(nChanges-1) = segments_iSpk(iSeg-1,3); % end time of the speaker's previous utterance
            changes(nChanges) = segments_iSpk(iSeg,2); % start time of the speaker's current utterance
        end
    end
    
    nChanges = nChanges + 1;
    changes(nChanges) = segments_iSpk(end,3); % end time of the speaker's last utterance
end

changes = sort(changes(1:nChanges));


end