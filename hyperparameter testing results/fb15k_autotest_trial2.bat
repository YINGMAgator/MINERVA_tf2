:: ensure that the code runs on GPU, where it can take advantage of the 400gb swap file
set CUDA_VISIBLE_DEVICES=""

:: block out warning messages that we're using almost all of the system's RAM and hard drive
set TF_CPP_MIN_LOG_LEVEL=2

:: Go to code package parent directory
cd C:\Users\owenb\OneDrive\Documents\GitHub\MINERVA_tf2

:: batch size 200, beta 0.2, lambda 0.0002, lr 1e-05
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 200 --beta 0.2 --Lambda 0.0002 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-5 --total_iterations=2000 --hp_type "combo trials" --hp_level "b0.2 lr1e-5 l0.0002 bs200"

:: we'll also try a very low learning rates with the high beta in case the divergence we see is because learning rate was too high in the first place
:: python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 2 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-5 --total_iterations=2000 --hp_type "combo trials" --hp_level "b2 lr1e-5"
:: python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 2 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-6 --total_iterations=2000 --hp_type "combo trials" --hp_level "b2 lr1e-6"

:: lambda 2 actually started by going down so we'll try that with low learning rate
:: python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 2 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-5 --total_iterations=2000 --hp_type "combo trials" --hp_level "l2 lr1e-5"

:: and with low learning rate AND high beta
:: python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 2 --Lambda 2 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-5 --total_iterations=2000 --hp_type "combo trials" --hp_level "l2 lr1e-5 b2"

:: and with high learning rate
:: python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 2 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-3 --total_iterations=2000 --hp_type "combo trials" --hp_level "l2 lr1e-3"

:: very high learning rate was interesting; maybe i can pair it with high entropy regularization and get a good result
:: python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 2 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-2 --total_iterations=2000 --hp_type "combo trials" --hp_level "lr1e-2 b2"

