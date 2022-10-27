function w2w2_combine_refs_for_multitask(outputName,varargin)
% combines single-task references for w2v2 audio classification into one file for multitask training
%
%   References must be json files in a very specific format (just having a valid json is not enough)
%      {"path":"/path/to/file1.wav","label":[0,0.1,0.2, ... ,0,0]}
%      {"path":"/path/to/file2.wav","label":[0,0,0, ... ,0.7,0.6]}
%   These files can be generated via prepare_data_for_wav2vec
%
% Usage:
%   w2w2_combine_refs_for_multitask(outputName, inputFile1, inputFile2, inputFile3, ...)
%   
%   outputName - path and filename for the output file
%   inputFile1, inputFile2, inputFile3, ... - paths to the input files
%   
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
% TODO: this takes a long time -> figure out a more efficient way (and ideally rewrite it in Python...)


%%

ref_files = varargin;

nFiles = numel(ref_files);

if nFiles < 2
    error('At least 2 input files are required');
end

lines_ref = cell(nFiles,1);

for iFile = 1:nFiles

    fileID = fopen(ref_files{iFile},'r');
    C = textscan(fileID,'%s','Delimiter','\n');
    fclose(fileID);

    lines_ref{iFile} = C{1};

    nLines = numel(lines_ref{iFile});


    if iFile == 1
        nLines = numel(lines_ref{iFile});
    elseif nLines ~= numel(lines_ref{iFile})
        warning('The number of records in all input files must be the same (first file: %d lines, file no. %d: %d lines)',...
        nLines, iFile, numel(lines_ref{iFile}));
    end

end

ref_filenames = cell(nLines,1);
ref_labels = cell(nLines,nFiles);

for iFile = 1:nFiles

    for iLine = 1:nLines

        str = strtrim(lines_ref{iFile}{iLine});

        idx = strfind(str,'","label":');
        if isempty(idx) || ~strcmp(str(1:9),'{"path":"') || str(end) ~= '}'
            error('Reference file must match a very specific format. See comments for details.')
            %      {"path":"/path/to/file1.wav","label":[0,0.1,0.2, ... ,0,0]}
            %      {"path":"/path/to/file2.wav","label":[0,0,0, ... ,0.7,0.6]}
        end

        filename = str(10:idx-1);


        if iFile == 1
            ref_filenames{iLine} = filename;
            loc = iLine;
        else
            [~,loc] = ismember(filename,ref_filenames);
            if loc < 1
                warning('''%s'' not found in ''%s''. It will be excluded.',filename,ref_files{1}); 
                continue;
            end
        end

        labels_str = str(idx+10:end-1);

        if labels_str(1) == '[' && labels_str(end) == ']'
            labels_str = labels_str(2:end-1);
        end

        C = strsplit(labels_str, ',');

        ref_labels{loc,iFile} = cell(size(C));
        for ii = 1:numel(C)
            ref_labels{loc,iFile}{ii} = C{ii};
        end
    end
end



fileID = fopen(outputName,'w');

for iLine = 1:nLines
    for iFile = 1:nFiles
        if isempty(ref_labels{iLine,iFile})
            warning('Missing labels for ''%s'' in file ''%s'', skipping.',iLine, ref_files{iFile});
            continue;
        end
    end

    % save as list of lists:
    %   [a1, a2, a3, ...], [b1,b2,b3,...] ->    "[[a1,b1],[a2,b2],[a3,b3],...]"
    fprintf(fileID,'{"path":"%s","label":[',ref_filenames{iLine});
    nFrames = length(ref_labels{iLine,1});
    for iFrame = 1:nFrames
        fprintf(fileID,'[%s',ref_labels{iLine,1}{iFrame});
        for iFile = 2:nFiles
            fprintf(fileID,',%s',ref_labels{iLine,iFile}{iFrame});
        end
        if iFrame < nFrames
            fprintf(fileID,'],');
        else
            fprintf(fileID,']');
        end
    end
    fprintf(fileID,']}\n');


end

fclose(fileID);


