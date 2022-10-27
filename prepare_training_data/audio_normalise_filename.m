function basename = audio_normalise_filename(wavname,dataset,suffixesToRemove,regexToRemove)
% obtains a normalised "base name" of an audio recording
%   by repeatedly removing specific substrings from the end (and only the end, to avoid false matches)
% e.g.:
%   'something_16kHz.stillPartOfTheName_16kHz.meeting.OHALLWAY'
%       will be turned into
%   'something_16kHz.stillPartOfTheName'
%
% version 2022-08-05
% 
% Usage:
%   basename = audio_normalise_filename(wavname,dataset,suffixesToRemove,regexToRemove)
%
%   dataset can be either one string (e.g. 'AMI') or a cell of strings (e.g. {'AMI', 'overlaps'})
%   
% ----
%
% Marie Kunesova (https://github.com/mkunes)
% 2022
%
% -----
%
% Changelog:
%   2022-10-27
%     - initial GitHub commit at https://github.com/mkunes/w2v2_audioFrameClassification/
%
% ---
% 
% TODO: all of this could probably be done in a much better way

if nargin < 3
    suffixesToRemove = {
        '.wav';
        '.mp4';
        '.dereverb';
        '.denoise';
%        '.16kHz';
        '.OMEETING';
        '.OHALLWAY';
        '.OOFFICE';
        '.meeting';
        '.booth';
        '.office';
%        '_16kHz';
        '_OMEETING';
        '_OHALLWAY';
        '_OOFFICE';
        '_meeting';
        '_booth';
        '_office';
    };    
end

if nargin < 4
    regexToRemove = {
        '(\.|_)\d+kHz'; % '.16kHz', '_8kHz', ...
    };
end

if ~iscell(dataset)
    dataset = {dataset};
end

if nargin < 2
    dataset = {'unknown'};
else
    if ismember('AMI', dataset)
        suffixesToRemove = [suffixesToRemove; '.Mix-Headset'];
    end
    
    if ismember('PROSYN-Rozhlas',dataset)
        suffixesToRemove = [suffixesToRemove; 
                '_all';
                '.perfect';
                ];
        regexToRemove = [regexToRemove;
            '\.sp\d+';
            '\.whn[0-9]+(\.[0-9]+)?';
            '_t[0-9]+(\.[0-9]+)?\-[0-9]+(\.[0-9]+)?';
        ];
    end

    if ismember('intervals',dataset)
        regexToRemove = [regexToRemove;
            '_t[0-9]+(\.[0-9]+)?\-[0-9]+(\.[0-9]+)?';
        ];
    end
    
    if ismember('augmented', dataset) || ismember('overlaps', dataset) || ismember('Libri', dataset)
        suffixesToRemove = [suffixesToRemove; 
            '.OMEETING';
            '.OHALLWAY';
            '.OOFFICE';
            '.meeting';
            '.booth';
            '.office';
            '_OMEETING';
            '_OHALLWAY';
            '_OOFFICE';
            '_meeting';
            '_booth';
            '_office';
            ];
        
        regexToRemove = [regexToRemove;
            '\.sp\d+';
            '\.whn[0-9]+(\.[0-9]+)?';
        ];
    end
end


basename = wavname;

regexStarts = [];
regexEnds = [];
for ii = 1:numel(regexToRemove)
   [startIndex,endIndex] = regexp(basename,regexToRemove{ii});
   regexStarts = [regexStarts, startIndex]; %#ok<AGROW>
   regexEnds = [regexEnds, endIndex]; %#ok<AGROW>
end

found = true;
while found
    found = false;
    for ii = 1:numel(suffixesToRemove)
        if endsWith(basename,suffixesToRemove{ii})
            basename = basename(1:(end-numel(suffixesToRemove{ii})));
            found = true;
        end
    end
    for ii = 1:numel(regexEnds)
       if regexEnds(ii) == numel(basename)
           basename = basename(1:regexStarts(ii)-1);
           found = true;
       end
    end
end
