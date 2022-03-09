:: ensure that the code runs on GPU, where it can take advantage of the 400gb swap file
set CUDA_VISIBLE_DEVICES=""

:: block out warning messages that we're using almost all of the system's RAM and hard drive
set TF_CPP_MIN_LOG_LEVEL=2

:: Go to code package parent directory
cd C:\Users\owenb\OneDrive\Documents\GitHub\MINERVA_tf2

:: BETA TESTING
:: 0.0002
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.0002 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-4 --total_iterations=300 --hp_type "beta" --hp_level "0.0002"

:: 0.002
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.002 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-4 --total_iterations=300 --hp_type "beta" --hp_level "0.002"

:: 0.02
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-4 --total_iterations=300 --hp_type "beta" --hp_level "0.02"

:: 0.2
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.2 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-4 --total_iterations=300 --hp_type "beta" --hp_level "0.2"

:: 2
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 2 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-4 --total_iterations=300 --hp_type "beta" --hp_level "2"
:: END OF BETA TESTING

:: LAMBDA TESTING
:: 0.0002
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.0002 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-4 --total_iterations=300 --hp_type "lambda" --hp_level "0.0002"

:: 0.002
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.002 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-4 --total_iterations=300 --hp_type "lambda" --hp_level "0.002"

:: 0.02
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-4 --total_iterations=300 --hp_type "lambda" --hp_level "0.02"

:: 0.2
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.2 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-4 --total_iterations=300 --hp_type "lambda" --hp_level "0.2"

:: 2
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 2 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-4 --total_iterations=300 --hp_type "lambda" --hp_level "2"
:: END OF LAMBDA TESTING


:: LEARNING RATE TESTING
:: 1e-2
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-2 --total_iterations=300 --hp_type "learning rate" --hp_level "1e-2"

:: 1e-3
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-3 --total_iterations=300 --hp_type "learning rate" --hp_level "1e-3"

:: 1e-4
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-4 --total_iterations=300 --hp_type "learning rate" --hp_level "1e-4"

:: 1e-5
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-5 --total_iterations=300 --hp_type "learning rate" --hp_level "1e-5"

:: 1e-6
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 256 --beta 0.02 --Lambda 0.02 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-6 --total_iterations=300 --hp_type "learning rate" --hp_level "1e-6"
:: END OF LEARNING RATE TESTING

:: LEARNING RATE TESTING COMPLETE