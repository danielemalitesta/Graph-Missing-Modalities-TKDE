from elliot.run import run_experiment
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--data', type=str, default='Beauty')
parser.add_argument('--model', type=str, default='mgcn')
parser.add_argument('--method', type=str, default='mean')
parser.add_argument('--layers', type=str, default='3')
parser.add_argument('--top_k', type=str, default='20')
parser.add_argument('--a', type=str, default='0.1')
parser.add_argument('--t', type=str, default='5.0')
parser.add_argument('--percentage', type=int, default=10)
parser.add_argument('--round', type=int, default=1)
args = parser.parse_args()

visual_folder_original_indexed = f'data/{args.data}/visual_embeddings_final_indexed_{args.percentage}_{args.round}'
textual_folder_original_indexed = f'data/{args.data}/textual_embeddings_final_indexed_{args.percentage}_{args.round}'

if args.method == 'heat':

    if not os.path.exists('./config_files/'):
        os.makedirs('./config_files/')
    if args.model == "mgcn":
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{args.data}/{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}/folder/recs/
      path_output_rec_weight: ./results/{args.data}/{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}/folder/weights/
      path_output_rec_performance: ./results/{args.data}/{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{args.data}/train_final_indexed.tsv
        validation_path: ../data/{args.data}/val_final_indexed.tsv
        test_path: ../data/{args.data}/test_final_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{args.data}/visual_embeddings_final_{args.method}_{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{args.data}/textual_embeddings_final_{args.method}_{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}_complete_indexed
      dataset: dataset_name
      top_k: 20
      evaluation:
        cutoffs: [20]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.MGCN:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr:  [ 0.0001, 0.001, 0.01 ]
          epochs: 200
          n_layers: 1
          n_ui_layers: 2
          top_k: 10
          factors: 64
          batch_size: 1024
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          normalize: True
          l_w: 1e-4
          c_l: [0.001, 0.01, 0.1]
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True 
            """
    elif args.model == "freedom":
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{args.data}/{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}/folder/recs/
      path_output_rec_weight: ./results/{args.data}/{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}/folder/weights/
      path_output_rec_performance: ./results/{args.data}/{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{args.data}/train_final_indexed.tsv
        validation_path: ../data/{args.data}/val_final_indexed.tsv
        test_path: ../data/{args.data}/test_final_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{args.data}/visual_embeddings_final_{args.method}_{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{args.data}/textual_embeddings_final_{args.method}_{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}_complete_indexed
      dataset: dataset_name
      top_k: 20
      evaluation:
        cutoffs: [20]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.FREEDOM:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          factors: 64
          epochs: 200
          l_w: [1e-5, 1e-2]
          n_layers: 1
          n_ui_layers: 2
          top_k: 10
          factors_multimod: 64
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          mw: (0.1,0.9)
          drop: 0.8
          lr_sched: (1.0,50)
          batch_size: 1024
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True
            """
    else:
        raise NotImplemented

    with open(f'./config_files/{args.model}_heat_{args.layers}_{args.top_k}_{args.t}_{args.data}_{args.percentage}_{args.round}.yml', 'w') as f:
        f.write(config.format(args.data).replace('dataset_name', args.data))

    visual_folder_imputed_indexed = f'./data/{args.data}/visual_embeddings_final_indexed_{args.method}_{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}'
    textual_folder_imputed_indexed = f'./data/{args.data}/textual_embeddings_final_indexed_{args.method}_{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}'
    visual_folder_complete = f'./data/{args.data}/visual_embeddings_final_{args.method}_{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}_complete_indexed'
    textual_folder_complete = f'./data/{args.data}/textual_embeddings_final_{args.method}_{args.percentage}_{args.round}_{args.layers}_{args.top_k}_{args.t}_complete_indexed'

    if not os.path.exists(visual_folder_complete):
        os.makedirs(visual_folder_complete)

    if not os.path.exists(textual_folder_complete):
        os.makedirs(textual_folder_complete)

    for it in os.listdir(visual_folder_original_indexed):
        shutil.copy(os.path.join(visual_folder_original_indexed, it), visual_folder_complete)
    for it in os.listdir(visual_folder_imputed_indexed):
        shutil.copy(os.path.join(visual_folder_imputed_indexed, it), visual_folder_complete)

    for it in os.listdir(textual_folder_original_indexed):
        shutil.copy(os.path.join(textual_folder_original_indexed, it), textual_folder_complete)
    for it in os.listdir(textual_folder_imputed_indexed):
        shutil.copy(os.path.join(textual_folder_imputed_indexed, it), textual_folder_complete)

    run_experiment(f"config_files/{args.model}_heat_{args.layers}_{args.top_k}_{args.t}_{args.data}_{args.percentage}_{args.round}.yml")

    shutil.rmtree(visual_folder_complete)
    shutil.rmtree(textual_folder_complete)

    os.remove(f"config_files/{args.model}_heat_{args.layers}_{args.top_k}_{args.t}_{args.data}_{args.percentage}_{args.round}.yml")

elif args.method == 'gae':

    if not os.path.exists('./config_files/'):
        os.makedirs('./config_files/')
    if args.model == "mgcn":
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{args.data}/{args.percentage}_{args.round}_{args.top_k}/folder/recs/
      path_output_rec_weight: ./results/{args.data}/{args.percentage}_{args.round}_{args.top_k}/folder/weights/
      path_output_rec_performance: ./results/{args.data}/{args.percentage}_{args.round}_{args.top_k}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{args.data}/train_final_indexed.tsv
        validation_path: ../data/{args.data}/val_final_indexed.tsv
        test_path: ../data/{args.data}/test_final_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{args.data}/visual_embeddings_final_{args.method}_{args.percentage}_{args.round}_{args.top_k}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{args.data}/textual_embeddings_final_{args.method}_{args.percentage}_{args.round}_{args.top_k}_complete_indexed
      dataset: dataset_name
      top_k: 20
      evaluation:
        cutoffs: [20]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.MGCN:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr:  [ 0.0001, 0.001, 0.01 ]
          epochs: 200
          n_layers: 1
          n_ui_layers: 2
          top_k: 10
          factors: 64
          batch_size: 1024
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          normalize: True
          l_w: 1e-4
          c_l: [0.001, 0.01, 0.1]
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True 
            """
    else:
        raise NotImplemented

    with open(f'./config_files/{args.model}_gae_{args.top_k}_{args.data}_{args.percentage}_{args.round}.yml', 'w') as f:
        f.write(config.format(args.data).replace('dataset_name', args.data))

    visual_folder_imputed_indexed = f'./data/{args.data}/visual_embeddings_final_indexed_{args.method}_{args.topk}_{args.percentage}_{args.round}'
    textual_folder_imputed_indexed = f'./data/{args.data}/textual_embeddings_final_indexed_{args.method}_{args.topk}_{args.percentage}_{args.round}'
    visual_folder_complete = f'./data/{args.data}/visual_embeddings_final_{args.method}_{args.percentage}_{args.round}_{args.top_k}_complete_indexed'
    textual_folder_complete = f'./data/{args.data}/textual_embeddings_final_{args.method}_{args.percentage}_{args.round}_{args.top_k}_complete_indexed'

    if not os.path.exists(visual_folder_complete):
        os.makedirs(visual_folder_complete)

    if not os.path.exists(textual_folder_complete):
        os.makedirs(textual_folder_complete)

    for it in os.listdir(visual_folder_original_indexed):
        shutil.copy(os.path.join(visual_folder_original_indexed, it), visual_folder_complete)
    for it in os.listdir(visual_folder_imputed_indexed):
        shutil.copy(os.path.join(visual_folder_imputed_indexed, it), visual_folder_complete)

    for it in os.listdir(textual_folder_original_indexed):
        shutil.copy(os.path.join(textual_folder_original_indexed, it), textual_folder_complete)
    for it in os.listdir(textual_folder_imputed_indexed):
        shutil.copy(os.path.join(textual_folder_imputed_indexed, it), textual_folder_complete)

    run_experiment(f"config_files/{args.model}_gae_{args.top_k}_{args.data}_{args.percentage}_{args.round}.yml")

    shutil.rmtree(visual_folder_complete)
    shutil.rmtree(textual_folder_complete)

    os.remove(f"config_files/{args.model}_gae_{args.top_k}_{args.data}_{args.percentage}_{args.round}.yml")

else:
    if not os.path.exists('./config_files/'):
        os.makedirs('./config_files/')
    if args.model == "freedom":
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{args.data}/{args.percentage}_{args.round}/folder/recs/
      path_output_rec_weight: ./results/{args.data}/{args.percentage}_{args.round}/folder/weights/
      path_output_rec_performance: ./results/{args.data}/{args.percentage}_{args.round}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{args.data}/train_final_indexed.tsv
        validation_path: ../data/{args.data}/val_final_indexed.tsv
        test_path: ../data/{args.data}/test_final_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{args.data}/visual_embeddings_final_{args.method}_{args.percentage}_{args.round}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{args.data}/textual_embeddings_final_{args.method}_{args.percentage}_{args.round}_complete_indexed
      dataset: dataset_name
      top_k: 20
      evaluation:
        cutoffs: [20]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.FREEDOM:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          factors: 64
          epochs: 200
          l_w: [1e-5, 1e-2]
          n_layers: 1
          n_ui_layers: 2
          top_k: 10
          factors_multimod: 64
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          mw: (0.1,0.9)
          drop: 0.8
          lr_sched: (1.0,50)
          batch_size: 1024
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True
            """
    elif args.model == "mgcn":
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{args.data}/{args.percentage}_{args.round}/folder/recs/
      path_output_rec_weight: ./results/{args.data}/{args.percentage}_{args.round}/folder/weights/
      path_output_rec_performance: ./results/{args.data}/{args.percentage}_{args.round}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{args.data}/train_final_indexed.tsv
        validation_path: ../data/{args.data}/val_final_indexed.tsv
        test_path: ../data/{args.data}/test_final_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{args.data}/visual_embeddings_final_{args.method}_{args.percentage}_{args.round}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{args.data}/textual_embeddings_final_{args.method}_{args.percentage}_{args.round}_complete_indexed
      dataset: dataset_name
      top_k: 20
      evaluation:
        cutoffs: [20]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.MGCN:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr:  [ 0.0001, 0.001, 0.01 ]
          epochs: 200
          n_layers: 1
          n_ui_layers: 2
          top_k: 10
          factors: 64
          batch_size: 1024
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          normalize: True
          l_w: 1e-4
          c_l: [0.001, 0.01, 0.1]
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True 
            """
    else:
        raise NotImplemented
    with open(f'./config_files/{args.model}_{args.method}_{args.percentage}_{args.round}.yml', 'w') as f:
        f.write(config.format(args.data).replace('dataset_name', args.data))

    visual_folder_imputed_indexed = f'./data/{args.data}/visual_embeddings_final_indexed_{args.method}_{args.percentage}_{args.round}'
    textual_folder_imputed_indexed = f'./data/{args.data}/textual_embeddings_final_indexed_{args.method}_{args.percentage}_{args.round}'
    visual_folder_complete = f'./data/{args.data}/visual_embeddings_final_{args.method}_{args.percentage}_{args.round}_complete_indexed'
    textual_folder_complete = f'./data/{args.data}/textual_embeddings_final_{args.method}_{args.percentage}_{args.round}_complete_indexed'

    if not os.path.exists(visual_folder_complete):
        os.makedirs(visual_folder_complete)

    if not os.path.exists(textual_folder_complete):
        os.makedirs(textual_folder_complete)

    for it in os.listdir(visual_folder_original_indexed):
        shutil.copy(os.path.join(visual_folder_original_indexed, it), visual_folder_complete)
    for it in os.listdir(visual_folder_imputed_indexed):
        shutil.copy(os.path.join(visual_folder_imputed_indexed, it), visual_folder_complete)

    for it in os.listdir(textual_folder_original_indexed):
        shutil.copy(os.path.join(textual_folder_original_indexed, it), textual_folder_complete)
    for it in os.listdir(textual_folder_imputed_indexed):
        shutil.copy(os.path.join(textual_folder_imputed_indexed, it), textual_folder_complete)

    run_experiment(f"config_files/{args.model}_{args.method}_{args.percentage}_{args.round}.yml")

    shutil.rmtree(visual_folder_complete)
    shutil.rmtree(textual_folder_complete)
