function prepare_data_for_wav2vec(wavListFile, dir_audio_out, dir_ref_in, dir_ref_out, task, varargin)
% prepares data for wav2vec2 finetuning and testing, for the tasks of 
%       a) overlapped speech detection (OSD)
%       b) voice activity detection (VAD)
%       c) speaker change detection (SCD)
%   (for wav2vec2_audioFrameClassification.py or wav2vec2_audioFrameClassification_multitask.py)
%
% - splits audio files into short segments, converting the sample rate if necessary 
%       (16 kHz by default, leave "newSampleRate" empty to disable conversion)
% - creates labels in a suitable format
%
% version 2022-10-27
%
% Marie Kunesova (https://github.com/mkunes)
% 2022
%
% ----
%
% Usage:
%   default settings:
%       prepare_data_for_wav2vec(wavListFile, dir_audio_out, dir_ref_in, dir_ref_out)
%   or with additional Name-Value Pair Arguments:
%       prepare_data_for_wav2vec(wavListFile, dir_audio_out, dir_ref_in, dir_ref_out, Name1, Value1, Name2, Value2, ...)
%
% wavListFile - path to a text file listing the wave files to process, one per line
%               lines can be commented-out using '#' or '%' at the start of the line
% dir_audio_out - target destination for the converted/segmented wav files
% dir_ref_in - directory with the ref. files corresponding to the wavs
%           either as a) .mat files with VAD/CF for both speakers,
%                  or b) classic RTTM files like for speaker diarization
%           if empty (dir_ref_in = []), script will instead look into the same folder as each wav file
% dir_ref_out - target destination for the reference labels
%
% task - 'OSD' / 'VAD' / 'SCD'
%
% ----
%
% Changelog:
%   2023-03-21
%     - replaced internal.stats.parseArgs with inputParser for compatibility with GNU Octave
%       (v6.4.0 is confirmed to work now; no other versions were tested)
%   2022-10-27
%     - combined OSD, VAD and SCD into the same main function
%     - initial GitHub commit at https://github.com/mkunes/w2v2_audioFrameClassification/
%
% ---
%
% TODO: add phrase boundary detection?
% TODO: allow creating reference labels for multiple tasks with one call
% TODO: OSD_VAD_generate_refs_for_wav2vec and SCD_generate_refs_for_wav2vec still have a lot of identical code
%       -> split it off into a separate function
% (Or even better, just rewrite everything in Python.)

options = {
    'maxTime', 20; % max audio duration, in seconds - recommended max. for wav2vec2 is 30s
    'minTime', 0.5; % minimum audio duration, in seconds - if the final segment would be shorter, it is discarded
    'shift', 10;   % when splitting longer wav files, shift by x seconds 
                % => if shift < maxTime, segments will partially overlap
                % if empty/unspecified, will be set to maxTime
    'dataset', 'unknown'; % identifier of the dataset, in case a specific dataset needs special handling
    'newSampleRate', 16000; % wavs will be converted to this sample rate. standard wav2vec2 requires 16kHz
    'labelsRate', 50; % how many labels per second there should be. standard wav2vec2 has 50 audio frames per second 

    'label_type', 'fuzzy'; %'binary'; % type of ref. labels: fuzzy = _/```\_, binary = _|``|_
    'label_collar', 0.2; %0; %0; %0.1; % width of the triangles in fuzzy references
    
    % min/max lengths of overlaps / non-overlaps, speech/non-speech
    % for OSD/VAD, the order of priority is: 
    %    1) fill in short gaps within the speech of the same speaker
    %    2) merge overlaps/speech intervals that are separated by very short non-overlaps/non-speech,
    %    3) remove very short overlaps/speech
    % SCD currently only uses minPauseLen_sameSpk
    % leave empty to use default settings:
    %   for SCD, the default is minPauseLen_sameSpk = 1; the other two are not used
    %   for VAD/OSD, the default is all zeroes
    'minPauseLen_sameSpk', []; % minimum pause within the speech of *the same speaker* - shorter pauses get relabeled as speech
                % out of these three settings, this is applied *first*
    'minNegLen', []; % minimum length of a non-overlap/non-speech between two 
                      % overlaps/speech intervals, in seconds - anything shorter gets relabeled
                % out of these three settings, this is applied *second*
                % (currently only used for VAD/OSD, has no effect on SCD)
    'minPosLen', []; % minimum length of an overlap/speech interval, in seconds - anything shorter gets relabeled
                % out of these three settings, this is applied *last*
                % (currently only used for VAD/OSD, has no effect on SCD)
    
                
    
    'ref_format', 'RTTM'; % a) 'RTTM' (classic NIST-style speaker diarization reference files)
                          % b) 'mat' - OSD/VAD only (.mat files with saved Matlab variables)
                          % c) 'none' (there are no refs available, the dataset is only for testing)
    'refFileSuffix', '.rttm'; % suffix of the reference files, e.g. '_labels.mat', '_RTTM.txt', '.rttm'
                                  % the names of the ref. files are expected as "<basename><suffix>" - e.g. "en_0638.rttm"
                                  
    'maxTotalTimePerOrigAudio', Inf; % maximum total duration of segments created from one long audio file, in seconds
                                     % e.g. a value of 60 means that only the first minute of each file will be used

    'splitWavsListFile', ''; % if not empty, the script will reuse existing split wavs, instead of creating new ones.
                             % the list of split wavs is given as a path to a text file (containing one path per line)

};

pnames = options(:,1);
dflts = options(:,2);

% Octave-compatible input parsing
p = inputParser;
for iArg = 1:numel(pnames)
    addParameter(p,pnames{iArg},dflts{iArg})
end
parse(p,varargin{:})

maxTime = p.Results.maxTime;
minTime = p.Results.minTime;
shift = p.Results.shift;
dataset = p.Results.dataset;
newSampleRate = p.Results.newSampleRate;
labelsRate = p.Results.labelsRate;
label_type = p.Results.label_type;
label_collar = p.Results.label_collar;
minPauseLen_sameSpk = p.Results.minPauseLen_sameSpk;
minNegLen = p.Results.minNegLen;
minPosLen = p.Results.minPosLen;
ref_format = p.Results.ref_format;
refFileSuffix = p.Results.refFileSuffix;
maxTotalTimePerOrigAudio = p.Results.maxTotalTimePerOrigAudio;
splitWavsList = p.Results.splitWavsListFile;

% % original parsing - not supported in Octave
% [maxTime,minTime,shift,dataset,newSampleRate,labelsRate,label_type,label_collar,...
%     minPauseLen_sameSpk,minNegLen,minPosLen,ref_format,refFileSuffix,...
%     maxTotalTimePerOrigAudio,splitWavsList] =  ...
%         internal.stats.parseArgs(pnames, dflts, varargin{1:end});
    
%%    

% split the audio files into segments of given length
if isempty(splitWavsList)
    splitWavsList = split_audio_for_wav2vec(wavListFile, dir_audio_out,...
        'maxTime',maxTime,'minTime',minTime,'shift',shift,'newSampleRate',newSampleRate,...
        'maxTotalTimePerOrigAudio',maxTotalTimePerOrigAudio);
end

%%

switch task
    case 'SCD' % speaker change detection
        
        % set task-specific defaults
        if isempty(minPauseLen_sameSpk)
            minPauseLen_sameSpk = 1;
        end
        if ~isempty(minNegLen) || ~isempty(minPosLen)
            warning('Settings ''minNonOLLen'' and ''minOLLen'' are currently not used for SCD - they will have no effect');
        end
        
        % generate reference labels for SCD
        SCD_generate_refs_for_wav2vec(splitWavsList, dir_ref_in, dir_ref_out, ...
            'dataset',dataset,'labelsRate',labelsRate,'label_type',label_type,'label_collar',label_collar,...
            'minPauseLen_sameSpk',minPauseLen_sameSpk,...
            'ref_format',ref_format,'refFileSuffix',refFileSuffix,'task',task);
        
    case {'OSD', 'VAD'} % overlapped speech and VAD use mostly the same code
        
        % set task-specific defaults
        if isempty(minPauseLen_sameSpk)
            minPauseLen_sameSpk = 0;
        end
        if isempty(minNegLen)
            minNegLen = 0;
        end
        if isempty(minPosLen)
            minPosLen = 0;
        end
        
        % generate reference labels for OSD or VAD
        OSD_VAD_generate_refs_for_wav2vec(splitWavsList, dir_ref_in, dir_ref_out, ...
            'dataset',dataset,'labelsRate',labelsRate,'label_type',label_type,'label_collar',label_collar,...
            'minPauseLen_sameSpk',minPauseLen_sameSpk,'minNonOLLen',minNegLen,'minOLLen',minPosLen,...
            'ref_format',ref_format,'refFileSuffix',refFileSuffix,'task',task);
    otherwise
        error('Invalid option: task == ''%s''',task);
end


    
