import os
import yaml
import logging
import json
import argparse
import torch

from src.datasets import dataset_factory
from src.training.backend import Config
from src.training.train import train, evaluate_datasets, safe_mean
from src.training.model import get_model_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_dir = os.path.join('configs', 'training')


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for the script')
    parser.add_argument(
        '--config', type=str, default=f'{config_dir}/example-config.yaml', help='Training config')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    training_run = False

    logger.info('Loading datasets...')
    if 'train' in config['dataset']:
        train_dataset = dataset_factory(**config['dataset']['train']).load()
        training_run = True
    
    dev_datasets = {}
    if 'dev' in config['dataset']:
        for dev_name, dev_config in config['dataset']['dev'].items():
            dev_datasets[dev_name] = dataset_factory(**dev_config).load()

    test_datasets = {}
    if 'test' in config['dataset']:
        for test_name, test_config in config['dataset']['test'].items():
            test_datasets[test_name] = dataset_factory(**test_config).load()
        
    logger.info('Loading training parameters...')
    train_cfg = Config()
    if 'train' in config['train_params']:
        for key, value in config['train_params']['train'].items():
            train_cfg.train.__setattr__(key, value)
            
    if 'optimizer' in config['train_params']:
        for key, value in config['train_params']['optimizer'].items():
            train_cfg.optimizer.__setattr__(key, value)
    
    if 'model' in config['train_params']:
        for key, value in config['train_params']['model'].items():
            train_cfg.model.__setattr__(key, value)
    
    if 'seed' in config['train_params']:
        train_cfg.seed = config['train_params']['seed']

    if 'name' in config['train_params']:
        train_cfg.name = config['train_params']['name']

    if 'train' in config['dataset'] and config['dataset']['train']['negatives']:
        train_cfg.negatives = config['dataset']['train']['negatives']

    run_folder = train_cfg.name if train_cfg.name != '' else train_cfg.timestamp
    run_path = os.path.join(train_cfg.train.output_path, run_folder)

    if training_run:
        logger.info('Commencing training...')
        train(train_cfg, train_dataset, dev_datasets, test_datasets)
        logger.info('Training finished.')
    else:
        logger.info('No training data provided. Skipping training...')
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        # save config
        with open(os.path.join(run_path, 'config.yaml') ,'w') as f:
            yaml.dump(train_cfg.to_dict(), f, default_flow_style=False)

    torch.cuda.empty_cache()

    # Load trained model
    logger.info('Loading model for evaluation...')
    model, tokenizer = get_model_tokenizer(train_cfg.model.name, 
                                           pooling=train_cfg.model.pooling,
                                           cache_dir=train_cfg.model.cache_dir)
    
    if os.path.exists(os.path.join(run_path, 'model.pt')):
        logger.info('Loading trained weights...')
        model.load_state_dict(torch.load(os.path.join(run_path, 'model.pt'), weights_only=True))
    
    model = model.to(train_cfg.train.device)
    
    # Evaluate the trained model
    logger.info('Evaluating the model...')
    results_path = os.path.join(run_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results = evaluate_datasets({**dev_datasets, **test_datasets},
                                model, tokenizer, results_path,
                                vectorizer_batch_size=train_cfg.model.vectorizer_batch_size, 
                                save_vectors=train_cfg.train.save_vectors,
                                device=train_cfg.train.device)
    
    mean_dev_ps = safe_mean([results[f'{dataset_name}_pair_success_at_10_mean'] for dataset_name in dev_datasets.keys()])
    mean_test_ps = safe_mean([results[f'{dataset_name}_pair_success_at_10_mean'] for dataset_name in test_datasets.keys()])
    logger.info(f'dev/mean-ps@10 {mean_dev_ps} test/mean-ps@10 {mean_test_ps}')

    logger.info('Saving the results...')
    with open(os.path.join(results_path, 'results.json'), 'w') as f:
        json.dump(results, f)

    # Uncomment to save disk space by deleting the final fine-tuned model
    #os.remove(os.path.join(run_path, 'model.pt'))  
