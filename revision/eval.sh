python -u eval.py --batch_size 16 \
  --dataset 'dcase' \
  --data_path '/mnt/shared/deep_learning/mwon/data' --model_load_path './../models/dcase_load_learn_6/best_model.pth' --input_length 48000 --n_fft 513 --conv_channels 128 --learn_f0 True 
