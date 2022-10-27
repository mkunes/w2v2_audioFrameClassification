MATLAB code for preparing training data for wav2vec2 audio frame classification

(Maybe I'll rewrite it in Python one day...)
    

How to use:

- create audio files and reference labels for single-task SCD, OSD or VAD:

    `prepare_data_for_wav2vec(wavListFile, dir_audio_out, dir_ref_in, dir_ref_out, task, <optional settings>)`

    where task = "SCD" / "OSD" / "VAD"

- combine single-task references into one file for multitask training:

    `w2w2_combine_refs_for_multitask(outputName, inputFile1, inputFile2, inputFile3, ...)`

   
- plot multitask predictions vs references:

    `w2v2_plot_outputs_vs_refs_multitask(file_results, file_ref, <optional settings>)`


