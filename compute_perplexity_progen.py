import torch
import sys
import pandas as pd
from src.baselines.datasets import ProGenDataset
from src.baselines.progen.modeling_progen import ProGenForCausalLM
from tqdm.auto import tqdm
import numpy as np
import argparse
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
url = "tcp://localhost:12355"
torch.distributed.init_process_group(backend="nccl", init_method = url, world_size=1, rank=0)



def load_progen_model(model_path = '../../checkpoints/progen-large'):


    # initialize the model with FSDP wrapper
    fsdp_params = dict(
        mixed_precision=True,
        flatten_parameters=True,
        state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
        cpu_offload=True,  # enable cpu offloading
    )

    with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        model = ProGenForCausalLM.from_pretrained(model_path)
        model.eval()

        # Wrap each layer in FSDP separately
        for name, child in model.named_children():
            if name == 'transformer':
                for name, child in child.named_children():
                    if name == "h":
                        for layer_name, layer in child.named_children():
                            wrapped_layer = wrap(layer)
                            setattr(child, layer_name, wrapped_layer)
        model = wrap(model)


    return model


def predict_progen(model, loader):

    with torch.no_grad():
        ppl = []
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            
            all_targets = batch.to(device)

            all_logits = model(all_targets, labels=all_targets).logits

            for idx in range(all_logits.shape[0]):
                logits = all_logits[idx]
                target = all_targets[idx]
                # unpad

                # shift
                logits = logits[:-1, ...]
                target = target[1:]

                # remove terminals
                bos_token, eos_token = 3, 4
                if target[-1] in [bos_token, eos_token]:
                    logits = logits[:-1, ...]
                    target = target[:-1]

                assert (target == bos_token).sum() == 0
                assert (target == eos_token).sum() == 0


                # remove unused logits
                first_token, last_token = 5, 29
                logits = logits[:, first_token:(last_token+1)]
                #import ipdb; ipdb.set_trace()
                target = target - first_token

                # note by computing this offset the padding token 0 will become -5
                loss = torch.nn.functional.cross_entropy(input=logits.cpu(), target=target.cpu(), ignore_index=-5, reduction='mean')
                ppl.append(torch.exp(loss).item())
    
    return ppl


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--skip_progen_fwd', action='store_true')
    parser.add_argument('--skip_progen_bwd', action='store_true')
    parser.add_argument('--data', type=str, default = '../data/fitness_data/grasso_prepared.csv')
    parser.add_argument('--progen_model_path', typ=str, default='../progen/')
    args = parser.parse_args()



    for progen_model in ['progen-small', 'progen-medium', 'progen-base', 'progen-large', 'progen-xlarge']:


        model =load_progen_model(os.path.join(args.progen_model_path, progen_model))

        if not args.skip_progen_fwd:

            out_dir = os.path.join(args.out_dir,f'{progen_model}_fwd')
            os.makedirs(out_dir, exist_ok=True)

            test_set = ProGenDataset(args.data,)
            
            loader = torch.utils.data.DataLoader(test_set, collate_fn = test_set.collate_fn, batch_size=10)

            perplexities = predict_progen(model, loader)
            df = test_set.df
            df['perplexity'] = perplexities
            df.to_csv(os.path.join(out_dir, args.data.split('/')[-1]))

        if not args.skip_progen_bwd:
            out_dir = os.path.join(args.out_dir,f'{progen_model}_bwd')
            os.makedirs(out_dir, exist_ok=True)

            test_set = ProGenDataset(args.data, reverse=True)
            
            loader = torch.utils.data.DataLoader(test_set, collate_fn = test_set.collate_fn, batch_size=10)

            perplexities = predict_progen(model, loader)

            df['perplexity'] = perplexities
            df.to_csv(os.path.join(out_dir, args.data.split('/')[-1]))



if __name__ == '__main__':
    main()