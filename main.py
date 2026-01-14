from model.build_freebase_dataset.get_gold_triplets import main as get_gold_triplets
from model.build_freebase_dataset.build_dataset import main as build_train_set
from model.message_passing.train_pipe import train
from model.inference.main_pure_graph_version import main as inference_main
from model.inference.evaluate import main as evaluate
from typing import Literal
import asyncio
import argparse

def pipe(dataset: Literal["webqsp", "cwq"]):
    max_concurrency = 10

    # build train_set
    asyncio.run(get_gold_triplets(save_path=f"./files/{dataset}/test_triplet.jsonl", ds_type="cwq", max_concurrency=max_concurrency))
    asyncio.run(build_train_set(save_path=f"./files/{dataset}/train.jsonl", ds_type="cwq", max_concurrency=max_concurrency))

    # train
    asyncio.run(train(ds_type=dataset))

    # inference
    asyncio.run(inference_main(ds_type=dataset,
                                     save_jsonl_path=f"./files/result/cwq/test.jsonl",
                                     max_concurrency=max_concurrency,
                                     feasible_threshold=0.1,
                                     quality_threshold=0.7))

    # evaluate
    evaluate(ds_type=dataset)




def main(argv=None):
    p = argparse.ArgumentParser(prog="app")
    p.add_argument("dataset", help="dataset (webqsp or cwq)")
    args = p.parse_args(argv)
    return pipe(args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())