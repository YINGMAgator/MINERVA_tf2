:: ensure that the code runs on GPU, where it can take advantage of the 400gb swap file
set CUDA_VISIBLE_DEVICES=""

:: block out warning messages that we're using almost all of the system's RAM and hard drive
set TF_CPP_MIN_LOG_LEVEL=2

:: Go to code package parent directory
cd C:\Users\owenb\OneDrive\Documents\GitHub\MINERVA_tf2

:: DEFAULTS
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0