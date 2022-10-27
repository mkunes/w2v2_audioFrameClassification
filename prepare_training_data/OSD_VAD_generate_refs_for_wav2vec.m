function OSD_VAD_generate_refs_for_wav2vec(splitWavsList, dir_ref_in, dir_ref_out, varargin)
% prepares reference labels for wav2vec2 finetuning and testing, for the task of overlapped speech detection or VAD
%       (for wav2vec2_audioFrameClassification.py or wav2vec2_audioFrameClassification_multitask.py)
%   WAV files must be already split into the desired length and must use the following name format:
%       "<original filename>_t<start time in seconds>_<end time in seconds>.wav"
%          e.g. "en_0638_t600-603.2.wav"
%       (technically, the wav files don't need to exist / be accessible; there just needs to be a list of them)
%   these split files and their list can be created via "split_audio_for_wav2vec" 
%       (or both can be called from "overlaps_prepare_data_for_wav2vec")
%
% Usage:
%   default settings:
%       overlaps_generate_refs_for_wav2vec(wavListFile, dir_ref_in, dir_ref_out)
%   or with additional Name-Value Pair Arguments:
%       overlaps_generate_refs_for_wav2vec(wavListFile, dir_ref_in, dir_ref_out, Name1, Value1, Name2, Value2, ...)
%
% wavListFile - path to a text file listing the segmented .wav files, one per line
%               lines can be commented-out using '#' or '%' at the start of the line
% dir_ref_in - directory with the ref. files corresponding to the wavs
%           either as a) .mat files with VAD/CF for both speakers,
%                  or b) classic RTTM files like for speaker diarization
%           
% dir_ref_out - target destination for the reference labels
%
% Marie Kunesova (https://github.com/mkunes)
% 2022
%
% ----
%
% Changelog:
%   2022-10-27
%     - initial GitHub commit at https://github.com/mkunes/w2v2_audioFrameClassification/
% ----

options = {
    'dataset', 'unknown'; % identifier of the dataset, in case a specific dataset needs special handling
    'labelsRate', 50; % how many labels per second there should be. standard wav2vec2 has 50 audio frames per second 

    'label_type', 'fuzzy'; %'binary'; %
    'label_collar', 0.2;
    
    % min/max lengths of overlaps / non-overlaps (for task == 'VAD', it's durations of speech / silence)
    % the order of priority is: 1) remove (relabel) short pauses within the speech of the same speaker
    %                           2) merge overlaps that are separated by very short non-overlaps,
    %                           3) remove very short overlaps
    'minPauseLen_sameSpk', 0; % minimum pause within the speech of *the same speaker* - shorter pauses get relabeled as speech
                % out of these three settings, this is applied *first*
    'minNonOLLen', 0; %0.5; % minimum length of a non-overlap between two overlaps, in seconds - anything shorter gets relabeled
                % out of these three settings, this is applied *second*
                % if task == 'VAD', this instead sets the minimum duration of *silence* between two speech intervals
    'minOLLen', 0.25; %0.5; % minimum length of an overlap, in seconds - anything shorter gets relabeled
                % out of these three settings, this is applied *last*
                % if task == 'VAD', this instead sets the minimum duration of a speech interval
    
    'ref_format', 'RTTM'; % reference format:
                          % a) 'RTTM' (classic NIST-style speaker diarization reference files)
                          % b) 'mat' (.mat files like in the old version of "overlaps_prepare_data_for_wav2vec")
                          % c) 'none' (there are no refs available, the dataset is only for testing)
    'refFileSuffix', '.rttm'; % suffix of the reference files, e.g. '_labels.mat', '_RTTM.txt', '.rttm'
                                  % the names of the ref. files are expected as "<basename><suffix>" - e.g. "en_0638.rttm"
    'task', 'OSD'; % 'OSD' / 'VAD'
};

pnames = options(:,1);
dflts = options(:,2);

[dataset,labelsRate,label_type,label_collar,minPauseLen_sameSpk,minNonOLLen,minOLLen,ref_format,refFileSuffix,task] =  ...
        internal.stats.parseArgs(pnames, dflts, varargin{1:end});
    
vps = 100; % sample rate of VAD/OSD in .mat reference files (not used with RTTM refs)
minNonOLLen_v = minNonOLLen * vps;
minOLLen_v = minOLLen * vps;

%% hardcoded settings

dataset_file_json = [dir_ref_out '/wav2vec2_refs_all.json']; % TODO - make configurable
dataset_file_txt = [dir_ref_out '/wav2vec2_refs_all.txt'];


%%

if ~exist(dir_ref_out,'dir')
    mkdir(dir_ref_out);
end

wavListFileID = fopen(splitWavsList, 'r');
lines = textscan(wavListFileID,'%s','delimiter','\n');
lines = lines{1};
numLines = length(lines);
fclose(wavListFileID);

segment_basenames = cell(numLines,1);
segment_times = zeros(numLines,2);
segment_fullpaths = cell(numLines,1);
    
nFiles = 0;
for iLine = 1:numLines
    
    wavFile = strtrim(lines{iLine});
    
    if isempty(wavFile) || wavFile(1) == '#' || wavFile(1) == '%'
        % skip commented-out lines
        continue;
    end
    nFiles = nFiles + 1;
    
    [~,segmentName,~] = fileparts(wavFile);
    basename = audio_normalise_filename(segmentName,{dataset, 'intervals'}); % "EN2002a.Mix-Headset_t0-10.wav" -> "EN2002a" etc.
    
    % get start and end time of the segment
    regex_str = '_t[0-9]+(\.[0-9]+)?\-[0-9]+(\.[0-9]+)?';
    [rs,re] = regexp(segmentName,regex_str);
    if rs > 0
        times_str = segmentName(rs+2:re);
        C = strsplit(times_str,'-');
        startTime = str2double(C{1});
        endTime = str2double(C{2});
    else
        warning('Failed to detect start and end times from filename ''%s'' - no match for regex ''%s''',segmentName, regex_str);
        startTime = -1;
        endTime = -1;
    end
    
    segment_basenames{nFiles} = basename;
    segment_fullpaths{nFiles} = wavFile;
    segment_times(nFiles,:) = [startTime, endTime];
    
end

segment_basenames = segment_basenames(1:nFiles);
segment_times = segment_times(1:nFiles,:);
segment_fullpaths = segment_fullpaths(1:nFiles);

%% reorder the arrays so that the split wavs are grouped by original file, then sorted by time

basenames_all = unique(segment_basenames);

seg_order = zeros(nFiles,1); % the new order of all segments (as reordered indices)
indices = 1:nFiles;
nOrdered = 0;

for ii = 1:numel(basenames_all)
    
    % sort only the segments that belong to <basename>
    idx = indices(strcmp(segment_basenames,basenames_all{ii}));
    [~,idx_order] = sort(segment_times(idx,1));
    idx = idx(idx_order);
    
    seg_order(nOrdered+1:nOrdered+numel(idx)) = idx;
    nOrdered = nOrdered + numel(idx);
end

% update the order of all relevant arrays
segment_basenames = segment_basenames(seg_order);
segment_fullpaths = segment_fullpaths(seg_order);
segment_times = segment_times(seg_order,:);

%% prepare the output files

for ii = 1:numel(basenames_all)
    
    % create an empty file <basename>.json
    fileID = fopen([dir_ref_out filesep basenames_all{ii} '.json'],'w'); 
    fclose(fileID);
end

FileIdDataset_json = fopen(dataset_file_json,'w');
FileIdDataset_txt = fopen(dataset_file_txt,'w');

%%

basename_prev = '';
for iFile = 1:nFiles

    [dir_wav, segmentName, ~] = fileparts(segment_fullpaths{iFile});
    basename = segment_basenames{iFile};
    
    %% get the ref. labels of the original unsplit file
    % if the new segment is from the same original file as the last one, we already have the labels - no need to redo them
    if ~strcmp(basename_prev,basename) 
        
        %-----------------
        % load the raw labels of the original unsplit file
        %----------------

        labels_filename = [dir_ref_in '/' basename refFileSuffix]; 
        
        switch ref_format
            case 'mat'
                labels_is_VAD = false;
                if ~exist(labels_filename, 'file')
                    labels_filename = [dir_ref_in filesep basename '_VAD.mat'];
                    if exist(labels_filename, 'file')
                        labels_is_VAD = true;
                    else
                        warning('Reference labels for file ''%s'' (basename ''%s'') not found!',segmentName, basename);
                        continue;
                    end
                end

                %labels = [];
                if labels_is_VAD
                    load(labels_filename,'VAD_both');
                    labels = sum(VAD_both, 2);
                else
                    load(labels_filename,'labels');
                end
                if isempty(labels)
                    warning('Labels for %s not found. Skipping.', segmentName);
                    continue;
                end

                switch task
                    case 'OSD'
                        overlaps = labels > 1;
                        overlaps = removeShortIntervals(overlaps,minNonOLLen_v,minOLLen_v);
                        intervals_times = getOverlapStartAndEndTimes(overlaps,vps); % get a Nx2 array of start and end times of overlaps
                    case 'VAD'
                        VAD = labels >= 1;
                        VAD = removeShortIntervals(VAD,minNonOLLen_v,minOLLen_v);
                        intervals_times = getOverlapStartAndEndTimes(VAD,vps); % get a Nx2 array of start and end times of overlaps
                    otherwise
                        error('Invalid option: task = ''%s''', task);
                end
            case 'RTTM'
                switch task
                    case 'OSD'
                        intervals_times = get_overlap_times_from_RTTM(labels_filename,minNonOLLen,minOLLen,minPauseLen_sameSpk);
                    case 'VAD'
                        intervals_times = get_vad_times_from_RTTM(labels_filename,minNonOLLen,minOLLen,minPauseLen_sameSpk);
                    otherwise
                        error('Invalid option: task = ''%s''', task);
                end
            case 'none'
                intervals_times = [];
        end

        %----------------
        % process the labels into the required format (still working with the full file)
        %---------------

        epsilon = 1e-10; % workaround for imperfect floating point comparisons

        audioEndTime = max(segment_times(strcmp(segment_basenames,basename),2));
        %audioEndTime = numel(data) / Fs;

        nFrames = ceil(audioEndTime * labelsRate - epsilon);
        labels_all = zeros(nFrames, 1);

        for iOverlap = 1:size(intervals_times,1)

            overlapStartTime = intervals_times(iOverlap, 1);
            overlapEndTime = intervals_times(iOverlap, 2);

            if strcmp(label_type, 'fuzzy')
                maxLabelStartTime = overlapStartTime + label_collar;
                maxLabelEndTime = overlapEndTime - label_collar;
            else
                maxLabelStartTime = overlapStartTime;
                maxLabelEndTime = overlapEndTime;
            end

            maxLabelStartIdx = round(maxLabelStartTime * labelsRate) + 1;
            maxLabelEndIdx = round(maxLabelEndTime * labelsRate);

            if maxLabelStartIdx > maxLabelEndIdx % the overlap is so short that the linear slopes can't reach the max value
                % => instead there will be just a smaller triangle with the peak in the middle

                % TODO: figure out if I shouldn't just ignore the overlap in this case
                maxLabelStartIdx = round(labelsRate * (overlapStartTime + overlapEndTime) / 2);
                maxLabelEndIdx = maxLabelStartIdx;
            end

            if strcmp(label_type, 'fuzzy') && label_collar > 0

                labelStartIdx = max(1,round((overlapStartTime - label_collar) * labelsRate));
                labelEndIdx = min(round((overlapEndTime + label_collar) * labelsRate),nFrames);

                % linear slope at the start of the overlap
                for idx = labelStartIdx:(maxLabelStartIdx-1)
                    if idx > nFrames
                        break;
                    end
                    time = idx / labelsRate;
                    label = 1 - abs(maxLabelStartTime - time) / label_collar;
                    label = round(label,4); % the rounding is mostly to avoid values like "5.68434e-14"
                    labels_all(idx) = max(labels_all(idx), label); 
                        % max is in case there are two overlaps close to each other
                end

                % linear slope at the end of the overlap
                for idx = (maxLabelEndIdx+1):labelEndIdx
                    if idx > nFrames
                        break;
                    end
                    time = idx / labelsRate;
                    label = 1 - abs(maxLabelEndTime - time) / label_collar;
                    label = round(label,4); % the rounding is mostly to avoid values like "5.68434e-14"
                    labels_all(idx) = max(labels_all(idx), label); 
                        % max is in case there are two overlaps close to each other
                end
            end

            % interval between the linear slopes gets label 1
            if isinf(maxLabelEndIdx)
                labels_all(maxLabelStartIdx:end) = 1;
            else
                labels_all(maxLabelStartIdx:maxLabelEndIdx) = 1;
            end

        end
        
    end
    basename_prev = basename;
    
    %% save the ref. labels for the specific interval we want
    
    startTime = segment_times(iFile,1);
    endTime = segment_times(iFile,2);

    labels_segment = labels_all(1+floor(startTime * labelsRate + epsilon):ceil(endTime * labelsRate - epsilon));

%     % text file with this split file's labels
%     fileID = fopen([dir_ref_out filesep segmentName '.txt'],'w');
%     fprintf(fileID,'%g ',labels_segment);
%     fprintf(fileID,'\n');
%     fclose(fileID);

    % json file with everything
    fprintf(FileIdDataset_json,'{"path":"%s","label":[%g',[dir_wav filesep segmentName '.wav'],labels_segment(1));
    if numel(labels_segment) > 1
        fprintf(FileIdDataset_json,',%g',labels_segment(2:end));
    end
    fprintf(FileIdDataset_json,']}\n');

    % <basename>.json - json lines only from this source wav
    fileID = fopen([dir_ref_out filesep basename '.json'],'a');
    fprintf(fileID,'{"path":"%s","label":[%g',[dir_wav filesep segmentName '.wav'],labels_segment(1));
    if numel(labels_segment) > 1
        fprintf(fileID,',%g',labels_segment(2:end));
    end
    fprintf(fileID,']}\n');
    fclose(fileID);

    % text file with everything
    fprintf(FileIdDataset_txt,'%s,[%g',[dir_wav filesep segmentName '.wav'],labels_segment(1));
    if numel(labels_segment) > 1
        fprintf(FileIdDataset_txt,' %g',labels_segment(2:end));
    end
    fprintf(FileIdDataset_txt,']\n');
        
        
    
end

fclose(FileIdDataset_json);
fclose(FileIdDataset_txt);

end

function overlaps_times = getOverlapStartAndEndTimes(overlaps, vps)
    changes = diff(overlaps);
    overlaps_times = zeros(ceil(sum(changes ~= 0) / 2), 2);
    
    nOverlaps = 0;
    isOverlap = false;
    
    if overlaps(1) == 1
        nOverlaps = 1;
        overlaps_times(1,1) = 0;
        isOverlap = true;
    end
    
    for idx = 1:numel(changes)
        switch changes(idx)
            case 0
                continue;
            case 1 % overlap starts at idx+1
                
                if isOverlap % sanity check
                    error('Unexpected value of changes(idx) - found "start of overlap" (value 1) before the previous one ended. This should never happen.');
                end
                    
                nOverlaps = nOverlaps + 1;
                overlaps_times(nOverlaps,1) = (idx+1) / vps;
                isOverlap = true;
            case -1 % overlap ends at idx+1
                
                if ~isOverlap % sanity check
                    error('Unexpected value of changes(idx) - found "end of overlap" (value -1) without a corresponding start. This should never happen.');
                end
                
                overlaps_times(nOverlaps,2) = (idx+1) / vps;
                isOverlap = false;
        end
    end
    if overlaps(end) == 1
        overlaps_times(nOverlaps,2) = Inf;  % if the end of the audio file has overlap, set the "end time" as Inf - 
                                            % we DON'T want the fuzzy labels to decrease here
    end
end

function overlaps = removeShortIntervals(overlaps_old,minNonOLLen_v,minOLLen_v)
% first, relabel non-overlaps shorter than minNonOLLen_v
% then, relabel overlaps shorter than minOLLen_v

    overlaps = overlaps_old;

    OLNow = overlaps(1);
    intStart = 1;
    for ii = 1:numel(overlaps)
        if OLNow
            if ~overlaps(ii)
                intStart = ii;
                OLNow = false;
            end
        else
            if overlaps(ii)
                if ii - intStart < minNonOLLen_v
                    % very short non-OL between OL -> relabel
                    overlaps(intStart:ii-1) = true;
                end
                intStart = ii;
                OLNow = true;
            end
        end
    end
    
    OLNow = overlaps(1);
    intStart = 1;
    for ii = 1:numel(overlaps)
        if OLNow
            if ~overlaps(ii)
                if ii - intStart < minOLLen_v
                    % very short OL between non-OL -> relabel
                    overlaps(intStart:ii-1) = false;
                end
                intStart = ii;
                OLNow = false;
            end
        else
            if overlaps(ii)
                intStart = ii;
                OLNow = true;
            end
        end
    end
end

