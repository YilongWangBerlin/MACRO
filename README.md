

## Models

Qwen/Qwen3-8B
google/gemma-2-9b-it

## 1,Preprocess
make prediction and target for gemma-2-9b-it

### 1.1 Make Predictions for gemma-2-9b-it

Terminal in /preprocess.
Run run_pred_all.sh
In addition, you may change the location of the model. 
SEE run_pred_all.sh

```
chmod +x run_pred_all.sh
./run_pred_all.sh
```

### 1.2 Make Targer for gemma-2-9b-it

run:
```
python3 make_target.py
```

You may check acc_calculate.ipynb to see the accuracy and align rate with en.

## 2. Sampling for Qwen3-8B

Terminal in /sample

run:
```
python generate_counterfactual.py   --pred_root ../data_qwen_pred   --out_root ../data_qwen_sample   --dataset_name sib200   --split train   --language zh   --max_samples 10   --model_name_or_path /root/autodl-tmp/model/Qwen3-8B
```

or run:
```
chmod +x run_sample_all.sh
./run_sample_all.sh
```

check model path

## 3. Reward Calculation

Terminal in /reward

run:
```
bash run_sample_all.sh
```

See reward distribution in /reward/test.ipynb