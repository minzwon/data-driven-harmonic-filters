python -u eval.py --batch_size 16 \
  --dataset 'mtat' \
  --data_path '/home/minz/Developer/data/mtat' --model_load_path './../models/model_melinit/best_model.pth' --learn_bw 'only_Q' --input_length 80000
