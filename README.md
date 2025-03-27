# Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation
 
This is the official implementation of the paper "_Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation_", currently under review.

## Requirements

Install the useful packages:

```sh
pip install -r requirements.txt
pip install -r requirements_torch_geometric.txt
```

## Datasets

### Datasets download

Download all the datasets from the original repository:

- Office:
  - Reviews: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products_5.json.gz
  - Metadata: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz
- Music:
  - Reviews: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz
  - Metadata: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Digital_Music.json.gz
- Baby:
  - Reviews: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz
  - Metadata: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Baby.json.gz
- Toys:
  - Reviews: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz
  - Metadata: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz
- Beauty: 
  - Reviews: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz
  - Metadata: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
- Sports:
  - Reviews: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz
  - Metadata: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz

And place each of them in the corresponding dataset folder accessible at ./data/<dataset-name>/. Then, run the following script:

```sh
python prepare_datasets.py --data <dataset_name>
```

this will create files for the items metadata and user-item reviews, and save the product images (check in the corresponding dataset folder). Moreover, statistics about the considered datasets will be displayed (e.g., missing modalities).


### Multimodal features extraction

After that, we need to extract the visual and textual features from items metadata and images. To do so, we use the framework [Ducho](https://github.com/sisinflab/Ducho), running the following configuration file for each dataset:

```yaml
dataset_path: ./data/<dataset_name>
gpu list: 0

visual:
    items:
        input_path: images
        output_path: visual_embeddings
        model: [
               { model_name: ResNet50,  output_layers: avgpool, reshape: [224, 224], preprocessing: zscore, backend: torch},
        ]

textual:
    items:
        input_path: final_meta.tsv
        item_column: asin
        text_column: description
        output_path: textual_embeddings
        model: [
            { model_name: sentence-transformers/all-mpnet-base-v2,  output_layers: 1, clear_text: False, backend: sentence_transformers},
          ]
```

where <dataset_name> should be substituted accordingly. This will extract visual and textual features for all datasets, accessible under each dataset folder.

### Missing features imputations

First, we perform imputation through traditional machine learning methods (zeros, random, mean). To do so, run the following script:

```sh
python run_split.py --data <dataset_name>
python impute.py --data <dataset_name> --gpu <gpu_id> --method <zeros_random_mean_ae>
```

this will create, under the specific dataset folder, an additional folder with **only** the imputed features, both visual and textual.

In a similar manner, we run the imputation through ae:

```sh
python impute_autoencoder.py --data <dataset_name> --gpu <gpu_id> --method ae
```

Before running the imputation through the graph-aware methods (gae, neigh_mean, feat_prop, pers_page_rank, heat), we need to split intro train/val/test and map all data processed so far to numeric ids. To do so, run the following script:

```sh

python to_id.py --data <dataset_name> --method <zeros_random_mean_ae>
```

this will create, for each dataset/modality/imputation folder, a new folder with the mapped (indexed) data. 

Now we can run the imputation with graph-aware methods.

```sh
python impute_all_multihop_pers_page_rank_heat.py --data <dataset_name> --method <method_name> --gpu <gpu_id>
chmod +777 impute_all_<method_name>_<dataset_name>.sh
./impute_all_<method_name>_<dataset_name>.sh
```

For gae, we run the following:

```sh
python impute_all_gae.py --data <dataset_name> --gpu <gpu_id>
chmod +777 impute_all_gae_<dataset_name>.sh
./impute_all_gae_<dataset_name>.sh
```

And for neigh_mean, we run:

```sh
python impute_all_neigh_mean.py --data <dataset_name> --gpu <gpu_id>
chmod +777 impute_all_neigh_mean_<dataset_name>.sh
./impute_all_neigh_mean_<dataset_name>.sh
```

Now we are all set to run the experiments. We use [Elliot](https://github.com/sisinflab/Formal-MultiMod-Rec) to train/evaluate the multimodal recommender systems.

### Results

#### Dropped setting

To obtain the results in the **dropped** setting, run the following scripts:
```sh
python run_split.py --data <dataset_name> --dropped True
python to_id_final.py --data <dataset_name>
python run_dropped.py --data <dataset_name>
```

#### Imputed setting

Then, we can compute the performance for the **imputed** setting. In the case of traditional machine learning imputation, we have:

```sh
python run_traditional_imputation.py --method <zeros_random_mean_ae> --dataset <dataset_name>
```

While, for other imputations, we run the following:

```sh
python run_multimodal.py --method <method_name> --dataset <dataset_name> --model <model_name> [--top_k <top_k> --layers <layers> --a <a> --t <t>]
# the parameters in [...] are optional and depend on the specific imputation method
```

#### Feature homophily calculation

To calculate feature homophily, run the following code:

```sh
python feature_homophily.py --data <dataset_name>
```

#### Percentages of missing modalities

First, mask some items at random with a certain percentage:

```sh
python mask_items.py --data <dataset_name> --percentages <percentages> --missing_modality <visual_or_textual_or_visual_textual>
```

Second, run the imputation. The scripts are the same as above, but terminate with "_percentages.py". You just need to specify the missing percentage as an additional input argument.

Finally, you may run the recommendation systems. You will first run for 0% of missing modalities:

```sh
python run_percentages.py --data <dataset_name>
```

And then, you will run:

```sh
python run_multimodal_percentages.py --percentage <percentage> [all arguments from run_multimodal.py]
```