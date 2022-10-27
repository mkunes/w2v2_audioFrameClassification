function w2v2_plot_outputs_vs_refs_multitask(file_results, file_ref, varargin)
% plots outputs from wav2vec2 audio frame classification
% file_results should be the original output file from w2v
% file_ref is a single JSON file containing the reference labels for each test utterance
%    it must have the same format as 'wav2vec_all.json', 
%      as generated in w2w2_combine_refs_for_multitask
%   (filenames can differ, but the format must be EXACTLY the same - a valid .json is NOT enough)
%   Expected .json format of ref. files:
%      {"path":"/path/to/file1.wav","label":[0,0.1,0.2, ... ,0,0]}
%      {"path":"/path/to/file2.wav","label":[0,0,0, ... ,0.7,0.6]}
%
% Marie Kunesova (https://github.com/mkunes)
% 2022
%
% ----
%
% Changelog:
%   2022-10-27
%     - initial GitHub commit at https://github.com/mkunes/w2v2_audioFrameClassification/
%
% ----
%
% TODO: accept any valid json file, not just this specific format
%

fig_ID = 22080901; % ID of the figure where everything will be plotted (just to avoid creating a new figure on every run)

%%

options = {
    'labelsRate', 50; % wav2vec labels per second (usually 50)
    'dataset', 'unknown'; %
    'ignoreEdges', 5; % ignore the first and last N seconds when plotting predictions
        % (dur 20s, shift 10s -> exclude 5s at the start and end of each audio file)
    'xLabelAsTime', true;
    'showSpeakers',false; % if true, shows intervals where each speaker is active 
        %   (via colored lines at the bottom of the last plot)
    'dir_RTTM', []; % only used if showSpeakers is true
    'suffix_RTTM', '.rttm'; % only used if showSpeakers is true
    
};

pnames = options(:,1);
dflts = options(:,2);

[labelsRate,dataset,ignoreEdges,xLabelAsTime,showSpeakers,dir_RTTM,suffix_RTTM] =  ...
        internal.stats.parseArgs(pnames, dflts, varargin{1:end});


%%

if nargin == 0
    
    dataset = 'AMI'; %
    
    % plot figure from the paper
    ignoreEdges = 0; % dur 20s, shift 10s -> exclude 5s at the start and end of each audio file 
    file_ref = 'examples/multitask_OSD_VAD_SCD/example_AMI_references.txt';
    file_results = 'examples/multitask_OSD_VAD_SCD/example_AMI_predictions.txt';
  
end


fileID = fopen(file_results,'r');
lines_results = textscan(fileID,'%s','Delimiter','\n');
fclose(fileID);

if isempty(file_ref)
    lines_ref = {};
else
    fileID = fopen(file_ref,'r');
    lines_ref = textscan(fileID,'%s','Delimiter','\n');
    fclose(fileID);
    
    lines_ref = lines_ref{1};
end

lines_results = lines_results{1};
nResults = numel(lines_results);

nRef = numel(lines_ref);

ref_filenames = cell(nRef,1);
ref_labels = cell(nRef,1);

nTasks = [];

for iLine = 1:nRef
    str = strtrim(lines_ref{iLine});
    
    idx = strfind(str,'","label":');
    if isempty(idx) || ~strcmp(str(1:9),'{"path":"') || str(end) ~= '}'
        error('Reference file must match a very specific format. See comments for details.')
        %   Expected .json format (for ref. files with ".json" extension):
        %      {"path":"/path/to/file1.wav","label":[0,0.1,0.2, ... ,0,0]}
        %      {"path":"/path/to/file2.wav","label":[0,0,0, ... ,0.7,0.6]}
        %   Expected text format (ref. files with any other extension):
        %      /path/to/file1.wav,[0 0.1 0.2 ... 0 0]
        %      /path/to/file2.wav,[0 0 0 ... 0.7 0.6]
    end
    [~,ref_filenames{iLine},~] = fileparts(str(10:idx-1));

    labels_str = str(idx+10:end-1);


    % note: using num2str() would be "easier", but it uses eval() -> avoid at all costs
    if labels_str(1) == '[' && labels_str(end) == ']'
        labels_str = labels_str(2:end-1);
    end
    if labels_str(1) == '[' && labels_str(end) == ']'
        labels_str = labels_str(2:end-1);
    end

    C = strsplit(labels_str, '],[');
    C2 = strsplit(C{1},',');
    
    nFrames = numel(C);
    
    if isempty(nTasks)
        nTasks = numel(C2);
    end
    
    ref_labels{iLine} = zeros(nFrames,nTasks);
    for ii = 1:nFrames
        C2 = strsplit(C{ii},',');
        if numel(C2) ~= nTasks
            error('Inconsistent number of labels per audio frame');
        end
        for jj = 1:nTasks
            ref_labels{iLine}(ii,jj) = str2double(C2{jj});
        end
    end

end

FigHandle = figure(fig_ID);
set(FigHandle, 'Position', [100, 100, 1200, 200 * nTasks]);


basename_prev = '';
isFirstFile = true;

for iLine = 1:nResults
    str = strtrim(lines_results{iLine});

    C = strsplit(str, ',');

    [~,filename,~] = fileparts(C{1});
    predictions_cell = C(2:end);
    
    basename = audio_normalise_filename(filename,{dataset,'intervals'});
    
    if strcmp(basename,basename_prev)
        isFirstFile = false;
        for iTask = 1:nTasks
            subplot(nTasks, 1, iTask);
            hold on;
        end
        
    else
        if ~isempty(basename_prev)
            subplot(nTasks, 1, 1);
            title(basename_prev, 'Interpreter','none');
            if showSpeakers
                file_RTTM = [dir_RTTM filesep basename_prev suffix_RTTM];
                if exist(file_RTTM,'file')
                    hold on;
                    plot_speakers(file_RTTM, labelsRate, xLabelAsTime);
                    hold off;
                else
                    warning('RTTM file ''%s'' not found. Speakers will not be shown.',file_RTTM);
                end
            end
            linkaxes;
            fprintf('Press any key to start plotting the next audio file.\n');
            pause;
        end
        for iTask = 1:nTasks
            subplot(nTasks, 1, iTask);
            cla;
        end
        basename_prev = basename;
        isFirstFile = true;
    end
    
    [~,loc] = ismember(filename,ref_filenames);
    
    regex_str = '_t[0-9]+(\.[0-9]+)?\-[0-9]+(\.[0-9]+)?';
    [rs,re] = regexp(filename,regex_str);
    if rs > 0
        times_str = filename(rs+2:re);
        C = strsplit(times_str,'-');
        startTime = str2double(C{1});
        %endTime = str2double(C{2});
    else
        warning('Failed to detect start and end times from filename ''%s'' - no match for regex ''%s''\n',filename, regex_str);
        startTime = 0;
        %endTime = -1;
    end
     
    if isempty(nTasks)
        nTasks = numel(predictions_cell);
    elseif numel(predictions_cell) ~= nTasks
        error('Number of tasks in ref. does not match the predictions');
    end
    
    for iTask = 1:numel(predictions_cell)
        subplot(nTasks, 1, iTask);
        predictions_str = predictions_cell{iTask};
        
        % note: using num2str() would be "easier", but it uses eval() -> avoid at all costs
        if predictions_str(1) == '[' && predictions_str(end) == ']'
            predictions_str = predictions_str(2:end-1);
        end
        
        C = strsplit(predictions_str, ' ');
        predictions = zeros(size(C));
        for ii = 1:numel(C)
            predictions(ii) = str2double(C{ii});
        end
        
        times = startTime + (1:numel(predictions)) ./ labelsRate;
        
        if xLabelAsTime
            times = seconds(times);
        end
        
        
        if ignoreEdges > 0
            ignoreEdges_L = round(ignoreEdges * labelsRate);
            if isFirstFile
                plot(times(1:end-ignoreEdges_L),predictions(1:end-ignoreEdges_L),'k-', 'LineWidth',2);
            else
                plot(times(ignoreEdges_L:end-ignoreEdges_L),predictions(ignoreEdges_L:end-ignoreEdges_L),'k-', 'LineWidth',2);
            end
        else
            %plot(times,predictions,'k:', 'LineWidth',2);
            plot(times,predictions,'k-', 'LineWidth',2);
        end
        
        if xLabelAsTime
            xtickformat('mm:ss')
        end
        
        hold on;
        
        
        if loc > 0
            times = startTime + (1:numel(ref_labels{loc}(:,iTask))) ./ labelsRate; % labels may be longer than predictions
            if xLabelAsTime
                times = seconds(times);
            end
            plot(times,ref_labels{loc}(:,iTask),'b', 'LineWidth',1)
        else
            fprintf('WARNING: filename not found among references: %s\n', filename);
        end
        xlabel('time [mm:ss]');
        ylim([-0.5 1.5]) 
    end
end

subplot(nTasks, 1, 1);
title(basename_prev, 'Interpreter','none');

if showSpeakers
    file_RTTM = [dir_RTTM filesep basename_prev suffix_RTTM];
    if exist(file_RTTM,'file')
        hold on;
        plot_speakers(file_RTTM, labelsRate, xLabelAsTime);
        hold off;
    else
        warning('RTTM file ''%s'' not found. Speakers will not be shown.',file_RTTM);
    end
end
linkaxes;



end

function plot_speakers(file_RTTM, labelsRate, xLabelAsTime)

    % read the entire RTTM
    fileID = fopen(file_RTTM, 'r');
    labels = textscan(fileID,'%s %s %d %f %f %s %s %s %[^\n\r]');
    % RTTM file entries are formatted like this:
    %   "SPEAKER" audio_ID channel_num start_time duration "<NA>" "<NA>" speaker_id [one or two more "<NA>"s]
    % e.g.:
    %   SPEAKER	2412-153948-0008_777-126732-0053	1	0.33	4.01	<NA>	<NA>	777	<NA>
    fclose(fileID);
    

    nSegments = numel(labels{1});
    speakers = unique(labels{8});
    nSpk = numel(speakers);
    
    % convert utterance start times and durations to indices of starts and ends
    start_idx = 1 + round(labels{4} * labelsRate);
    end_idx = start_idx + round(labels{5} * labelsRate);
    
    nFrames = ceil(max(end_idx));
    
    spk_VAD = zeros(nFrames,nSpk);
    
    
    for iSeg = 1:nSegments
        if ~strcmp(labels{1}{iSeg},'SPEAKER')
            continue;        
        end
        spkName = labels{8}{iSeg};
        [~,spkId] = ismember(spkName,speakers);
        
        spk_VAD(start_idx(iSeg):end_idx(iSeg), spkId) = 1;
    end
    
    times = (1:nFrames) ./ labelsRate;
    if xLabelAsTime
        times = seconds(times);
    end

    for iSpk = 1:nSpk
        plot(times,spk_VAD(:,iSpk) * (-0.05 * iSpk),'.');
    end



end