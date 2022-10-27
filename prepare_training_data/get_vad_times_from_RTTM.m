function vad_times = get_vad_times_from_RTTM(rttm_filename,minSilLen,minSpeechLen,minPauseLen_sameSpk)
% get VAD from RTTM, as a Nx2 array of speech start times and end times
%
% ---
%
% Marie Kunesova (https://github.com/mkunes)
% 2022


% these arguments are a holdover from "get_overlap_times_from_RTTM"
%   but for VAD, all of them should probably stay as 0
if nargin < 4
    minPauseLen_sameSpk = 0;
    
    if nargin < 3
        minSpeechLen = 0;
        
        if nargin < 2
            minSilLen = 0;
        end
    end
end


% read the entire RTTM
fileID = fopen(rttm_filename, 'r');
labels = textscan(fileID,'%s %s %d %f %f %s %s %s %[^\n\r]');
    % RTTM file entries are formatted like this:
    %   "SPEAKER" audio_ID channel_num start_time duration "<NA>" "<NA>" speaker_id [one or two more "<NA>"s]
    % e.g.:
    %   SPEAKER	2412-153948-0008_777-126732-0053	1	0.33	4.01	<NA>	<NA>	777	<NA>
fclose(fileID);

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

% generate a list of speech intervals by tracking the number of active speakers after each speaker change
vad_times = zeros(size(changes));
active_speakers = [];
nSpeech = 0;
isSpeech = false;

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
            if numel(active_speakers) > 0 && ~isSpeech
                % speech starts here
                isSpeech = true;
                
                if nSpeech == 0 || t - vad_times(nSpeech,2) >= minSilLen
                    % it the time since last speech is long enough, add this as a new speech interval
                    %   (otherwise they get merged)
                    
                    nSpeech = nSpeech + 1;
                    vad_times(nSpeech,1) = changes(iChng,1);
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
            if numel(active_speakers) < 1 && isSpeech
                % speech ends here
                isSpeech = false;
                vad_times(nSpeech,2) = changes(iChng,1);
            end
        end
    end
end
if isSpeech
    vad_times(nSpeech,2) = Inf;
end
vad_times = vad_times(1:nSpeech,:);


durations = vad_times(:,2) - vad_times(:,1); % duration of each speech interval
vad_times = vad_times(durations >= minSpeechLen,:); % only keep speech intervals that are long enough

end

