#Multi-target Voice Conversion without Parallel Data by Adverserially Learning Disentangled Audio Representations

Steps for training:
1.) Use make_dataset_vctk.py to generate data from VCTK Corpus which is used as data during training and testing.

    Usage: python3.6 make_dataset_vctk.py --N_speakers 20 --dataset_output ./vctk_20.h5 --vctk_info /data/speech/VCTK-Corpus/speaker-info.txt --vctk_wav /data/speech/VCTK-Corpus/wav16/'
    
    Output: 
    - h5py (HDFS) file containing linear spectrogram(lin), mel-spectogram(mel), logarithmic fundamental frequency (log_f0) and mel-cepstrum (mcep) for each utterance by each of the N_speakers sampled randomnly from VCTK speakers pool.
        # Order of data in dataset.h5py file: 
            dataset
            |-- train 
            |    |--- speaker_id
            |           |---- lin
            |           |---- mel
            |           |---- log_f0
            |           |---- mc
            |-- test
                 |--- speaker_id
                        |---- lin
                        |---- mel
                        |---- log_f0
                        |---- mc
    - txt file with speaker_id and gender of speaker for each speaker sampled for dataset generated.
        # Example: 
          225F
          226M
        # File format: Each line represents {speaker_id}{gender}
                       
2.) Use make_single_samples.py to make the sampling index used in training to use data from above.
    
    
    Usage: python3.6 make_single_samples.py --dataset_path ./vctk_20.h5 --index_output vctk_20_index.json
    

3.) Train using main.py
      
    
    CUDA_AVAILABLE_DEVICES=0,1 python3.6 main.py -dataset_path /data/speech/processed/vctk/vctk_random20.h5 -index_path /data/speech/processed/vctk/vctk_random20_index.json -output_model_path /data/speech/processed/vctk/models/model.pkl > /logs/vctk_vc.txt &

4.) Generate converted samples using convert.py. Two modes are available: Single Pair and All Pairs conversion.

    Usage ( all pairs ) : python3.6 convert.py --results_dir ./converted_audio --dataset_path ./vctk_20.h5 --model_path ./.model_single_sample.pkl-1 --speakers_used ./vctk_20_speakers_used.txt --all True
    Usage ( single pair ) : python3.6 convert.py --results_dir ./converted_audio --dataset_path ./vctk_20.h5 --model_path ./.model_single_sample.pkl-1 --speakers_used ./vctk_20_speakers_used.txt --single True --source  261 --target 259
    
    Output:
    The folder are appened with gender:
    Example: p261F_259M
             |
             |------261F_259M_{utt_id}.wav
                    .
                    .
                    .
    






