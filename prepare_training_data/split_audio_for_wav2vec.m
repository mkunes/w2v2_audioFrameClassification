function wav_list_out = split_audio_for_wav2vec(wavListFile, dir_audio_out, varargin)
% splits long audio files into shorter WAVs of the specified length, with optional overlap,
% 	converting the sample rate if necessary (16 kHz by default, leave "newSampleRate" empty to disable conversion)
% the resulting short files can be used for e.g. wav2vec
%
% version 2022-10-05
%
% Usage:
%   default settings:
%       split_audio_for_wav2vec(wavListFile, dir_audio_out)
%   or with additional Name-Value Pair Arguments:
%       split_audio_for_wav2vec(wavListFile, dir_audio_out, Name1, Value1, Name2, Value2, ...)
%
% wavListFile - path to a text file listing the wave files to process, one per line
%               lines can be commented-out using '#' or '%' at the start of the line
% dir_audio_out - target destination for the converted/segmented wav files
%
%
% Marie Kunesova (https://github.com/mkunes)
% 2022
%
% ----
%
% Changelog:
%   2023-03-21
%     - replaced internal.stats.parseArgs with inputParser for compatibility with GNU Octave
%       (v6.4.0 is confirmed to work now; no other versions were tested)
%   2022-10-27
%     - initial GitHub commit at https://github.com/mkunes/w2v2_audioFrameClassification/
%
% ----
%
% TODO: Check if sox is available; if not, use built-in Matlab functions for sample rate conversion
%   (or even better, rewrite this entire thing in Python)
%

options = {
    'maxTime', 20; % max audio duration, in seconds - recommended max. for wav2vec2 is 30s
    'minTime', 0.5; % minimum audio duration, in seconds - if the final segment would be shorter, it is discarded
    'shift', 10; % when splitting longer wav files, shift by x seconds 
                % => if shift < maxTime, segments will partially overlap
                % if empty, will be set to maxTime
    'newSampleRate', 16000; % wavs will be converted to this sample rate. standard wav2vec2 requires 16kHz
                        % leave empty to keep the original sample rate
    'maxTotalTimePerOrigAudio', Inf; % maximum total duration of each long audio file to be used, in seconds
                                     % e.g. a value of 60 means that only the first minute is used
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
newSampleRate = p.Results.newSampleRate;
maxTotalTimePerOrigAudio = p.Results.maxTotalTimePerOrigAudio;

% % original parsing - not supported in Octave
% [maxTime,minTime,shift,newSampleRate,maxTotalTimePerOrigAudio] =  ...
%         internal.stats.parseArgs(pnames, dflts, varargin{1:end});

dir_out_main = dir_audio_out;


%%

% sanity checks
if maxTime <= 0
    error('maxTime must be > 0');
end

if isempty(shift)
    shift = maxTime;
elseif shift <= 0 || shift > maxTime
    error('shift must be > 0 and <= maxTime');
end


%%

if ~exist(dir_audio_out,'dir')
    mkdir(dir_audio_out);
end

tempFile = [dir_audio_out filesep 'temp.wav']; % temporary file used for sample rate conversion


wavListFileID = fopen(wavListFile, 'r');
lines = textscan(wavListFileID,'%s','delimiter','\n');
lines = lines{1};
numLines = length(lines);
fclose(wavListFileID);


wav_list_out = [dir_out_main filesep 'list_wav_segments.txt'];
FileId = fopen(wav_list_out,'w');

iFile = 0;
for iLine = 1:numLines
    
    wavFile = strtrim(lines{iLine});
    
    if isempty(wavFile) || wavFile(1) == '#' || wavFile(1) == '%'
        % skip commented-out lines
        continue;
    end
    iFile = iFile + 1;

    [data, Fs] = audioread(wavFile);
    
    %% convert sample rate if needed
    % TODO: use Matlab built-in functions if sox is not available
    
    if ~isempty(newSampleRate) && Fs ~= newSampleRate % sample rate doesn't match the target
        % -> create a temporary file with the desired sample rate, using sox
        %   (yes, I could resample it via built-in Matlab functions, but sox is probably better)
        
        sysString = sprintf('sox "%s" -r %d "%s"',wavFile,newSampleRate,tempFile);
        system(sysString);
        
        [data, Fs] = audioread(tempFile);
        delete(tempFile);
        
    end
    
    %% split audio into chunks of set length
    
    segmentStartTime = 0;
    audioEndTime = min(numel(data) / Fs,maxTotalTimePerOrigAudio);
    
    [~, name, ~] = fileparts(wavFile);
    
    epsilon = 1e-10; % workaround for imperfect floating point comparisons
    
    while segmentStartTime + minTime < audioEndTime
        
        segmentEndTime = min(segmentStartTime + maxTime,audioEndTime);
        newName = sprintf('%s_t%g-%g', name, segmentStartTime, segmentEndTime);

        segment_fullpath = [dir_audio_out filesep newName '.wav'];
        
        startIdx = 1 + floor(segmentStartTime * Fs+epsilon);
        endIdx = min(ceil(segmentEndTime * Fs - epsilon),size(data,1));
        
        audiowrite(segment_fullpath, data(startIdx:endIdx), Fs);
        fprintf(FileId,'%s\n',segment_fullpath);
        
        segmentStartTime = segmentStartTime + shift;
        
        if segmentEndTime == audioEndTime
            % the segment that was just created already includes the end of the original audio
            %   -> there's no need to make one more segment that's just a subset of this
            break;
        end
    end
end

fclose(FileId);

end



