function SCD_generate_refs_for_wav2vec(splitWavsList, dir_ref_in, dir_ref_out, varargin)
% prepares reference labels for wav2vec2 finetuning and testing, for the task of speaker change detection
%       (for wav2vec2_audioFrameClassification.py or wav2vec2_audioFrameClassification_multitask.py)
%   WAV files must be already split into the desired length and must use the following name format:
%       "<original filename>_t<start time in seconds>_<end time in seconds>.wav"
%          e.g. "en_0638_t600-603.2.wav"
%       (technically, the wav files don't need to exist / be accessible; there just needs to be a list of them)
%   these split files and their list can be created via "split_audio_for_wav2vec" 
%       (or both can be called from "SCD_prepare_data_for_wav2vec")
%
% Usage:
%   default settings:
%       SCD_generate_refs_for_wav2vec(wavListFile, dir_ref_in, dir_ref_out)
%   or with additional Name-Value Pair Arguments:
%       SCD_generate_refs_for_wav2vec(wavListFile, dir_ref_in, dir_ref_out, Name1, Value1, Name2, Value2, ...)
%
% wavListFile - path to a text file listing the segmented .wav files, one per line
%               lines can be commented-out using '#' or '%' at the start of the line
% dir_ref_in - directory with the ref. files corresponding to the wavs, in RTTM format
%           
% dir_ref_out - target destination for the reference labels
%
% Marie Kunesova (https://github.com/mkunes)
% 2022
%
% ----
%
% Changelog:
%   2023-03-21
%       - replaced internal.stats.parseArgs with inputParser for compatibility with GNU Octave
%       (v6.4.0 is confirmed to work now; no other versions were tested)
%       - resolved a slight rounding inconsistency caused by Matlab's imperfect floating point arithmetic
%           (in the part where "the exact frame gets a 1, even with fuzzy labelling")
%   2023-02-20
%       - round labels to 4 decimal places (to avoid "2.22045e-16" etc)  
%   2022-10-27
%     - initial GitHub commit at https://github.com/mkunes/w2v2_audioFrameClassification/
%
% ----
%
% TODO: if two different speakers have a pause or overlap between them that's shorter than X seconds, 
%           mark it as only ONE change, rather than 2


options = {
    'dataset', 'unknown'; % identifier of the dataset, in case a specific dataset needs special handling
    'labelsRate', 50; % how many labels per second there should be. standard wav2vec2 has 50 audio frames per second 

    'label_type', 'fuzzy'; %'binary'; %
    'label_collar', 0.2;
    
    'minPauseLen_sameSpk', 1; % minimum pause within the speech of *the same speaker* - shorter pauses get relabeled as speech
                % out of these three settings, this is applied *first*
    
    'ref_format', 'RTTM'; % reference format:
                          % a) 'RTTM' (classic NIST-style speaker diarization reference files)
                          % b) 'none' (there are no refs available, the dataset is only for testing)
        
    'refFileSuffix', '.rttm'; % suffix of the reference files, e.g. '_labels.mat', '_RTTM.txt', '.rttm'
                                  % the names of the ref. files are expected as "<basename><suffix>" - e.g. "en_0638.rttm"
    'task', 'SCD';
};

pnames = options(:,1);
dflts = options(:,2);

% Octave-compatible input parsing
p = inputParser;
for iArg = 1:numel(pnames)
    addParameter(p,pnames{iArg},dflts{iArg})
end
parse(p,varargin{:})

dataset = p.Results.dataset;
labelsRate = p.Results.labelsRate;
label_type = p.Results.label_type;
label_collar = p.Results.label_collar;
minPauseLen_sameSpk = p.Results.minPauseLen_sameSpk;
ref_format = p.Results.ref_format;
refFileSuffix = p.Results.refFileSuffix;
task = p.Results.task;

% % original parsing - not supported in Octave
%[dataset,labelsRate,label_type,label_collar,minPauseLen_sameSpk,ref_format,refFileSuffix,task] =  ...
%        internal.stats.parseArgs(pnames, dflts, varargin{1:end});
    
if exist('OCTAVE_VERSION', 'builtin') ~= 0 % Octave
    isOctave = true;
else
    isOctave = false;
end

%% hardcoded settings

dataset_file_json = [dir_ref_out '/wav2vec2_refs_all.json']; % TODO - make configurable
dataset_file_txt = [dir_ref_out '/wav2vec2_refs_all.txt'];

minPauseLen_start_end = minPauseLen_sameSpk; % if the pause at the start or end of the audio file is shorter than this, 
                             %  don't count the start of the first utterance / end of the last one as a "speaker change"
                             % TODO: this value should probably be smaller than minPauseLen_sameSpk

%%

if ~exist(dir_ref_out,'dir')
    mkdir(dir_ref_out);
end


%% read the list of split wavs, determine the original filenames and times

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
        
        % end time of the full, unsplit file
        audioEndTime = max(segment_times(strcmp(segment_basenames,basename),2));
        
        %-----------------
        % load the raw labels of the original unsplit file
        %----------------

        switch ref_format
            case 'RTTM'
                labels_filename = [dir_ref_in '/' basename refFileSuffix];

                switch task
                    case 'SCD'
                        changes_ref = get_speaker_changes_from_RTTM(labels_filename,minPauseLen_sameSpk);
                        
                        % if there's no/very small pause at the start or end of the audio file, 
                        %  don't count the start of the first utterance / end of the last one as a "speaker change"
                        if changes_ref(1) < minPauseLen_start_end 
                            changes_ref = changes_ref(2:end);
                        end
                        if audioEndTime - changes_ref(end) < minPauseLen_start_end
                            changes_ref = changes_ref(1:end-1);
                        end
                        
                    otherwise
                        error('Invalid option: task = ''%s''', task);
                end
            case 'none'
                changes_ref = [];
        end

        %----------------
        % process the labels into the required format (still working with the full file)
        %---------------

        epsilon = 1e-10; % workaround for imperfect floating point comparisons

        nFrames = ceil(audioEndTime * labelsRate - epsilon);
        labels_all = zeros(nFrames, 1);

        for iChange = 1:numel(changes_ref)

            labelMidTime = changes_ref(iChange);

            if labelMidTime == 0
                continue;
            end

            if label_collar > 0

                labelStartIdx = max(1,round((labelMidTime - label_collar) * labelsRate));
                labelEndIdx = min(nFrames,round((labelMidTime + label_collar) * labelsRate));

                switch label_type
                    case 'fuzzy'
                        for idx = labelStartIdx:labelEndIdx
                            time = idx / labelsRate;
                            label = 1 - abs(labelMidTime - time) / label_collar;
                            
                            % round to 4 decimals (mostly to avoid values like "5.68434e-14")
                            if isOctave % Octave 6.4.0 does not support rounding to N decimals using round(X,N)
                                k = 10000;
                                label = round(label * k) / k;
                                
                                % this seemingly pointless bit of code gets rid of "negative zeros"
                                %  - Octave 6.4.0 prints them as "-0" when using %g
                                if label == 0 
                                    label = 0;
                                end
                            else
                                label = round(label,4); 
                            end
                            
                            labels_all(idx) = max(labels_all(idx), label); 
                                % max is in case there are two refs close to each other
                        end
                    case 'binary'
                        labels_all(labelStartIdx:labelEndIdx) = 1;
                    otherwise
                        error('Invalid option: labels_type = ''%s''',labels_type);
                end
            end

            % the exact frame gets a "1", even with fuzzy labelling
            % edit 2023-03-21: added epsilon for rounding consistency:
            %    if the real speaker change is exactly between two wav2vec frames, now it should always get rounded to the second one
            %    previously, it could go either way (Matlab's imperfect floating point arithmetic again...)
            labels_all(round(labelMidTime * labelsRate + epsilon)) = 1; 
                
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
