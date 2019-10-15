python -u eval.py --batch_size 16 \
  --dataset 'mtat' \
  --data_path '/home/minz/Developer/data/mtat' --model_load_path './../models/best_model.pth' --input_length 80000 --conv_channels 128 --learn_f0 True --learn_bw True --semitone_scale 2
