

from typing import Callable, Iterator, AsyncIterator, Literal
from model.vocab.knowledge_graph_vocab import KnowledgeGraphVocab
import json
from model.schemas import KGDataset
from config.model_args import DEVICE
from model.vocab.freebase_vocab import FreebaseVocab

freebase_vocab = FreebaseVocab()

def graph_dataset_iterator(ds_type: Literal["cwq", "webqsp"], device: str = DEVICE, mode: Literal["train", "validate"]= "train", val_size: float = 0.05) -> Iterator:
    """
    一个迭代器：逐行读取 jsonl 文件，解析为 dict，并执行用户指定的操作。

    参数:
        path: jsonl 文件路径
        op:   用户定义的函数，接收当前行的 dict 作为输入

    返回:
        每一行的 dict (yield 出来，供外部遍历使用)
    """
    if ds_type == "cwq":
        ds_path = "./files/CWQ/train.jsonl"
    elif ds_type == "webqsp":
        ds_path = "./files/WebQSP_official/processed_with_text.jsonl"
    else:
        raise ValueError()
    with open(ds_path, "r", encoding="utf-8") as f:
        s = sum(1 for _ in f)
        f.seek(0)

        for idx, line in enumerate(f):
            if mode == "validate" and idx < s*(1 - val_size):
                continue
            elif mode == "train" and idx >= s*(1 - val_size):
                break

            line = line.strip()
            if not line:
                continue
            data = json.loads(line)  # 转为 dict

            if data['source_entity_text'] is None:
                continue
            if None in data['answer_entity_text']:
                continue
            for t in data['full_text_triplet']:
                if None in t:
                    continue

            kg = KGDataset(
                device = device,
                triplet_text=data['full_text_triplet'],
                query=data['question'],
                source_entity=[data['source_entity_text']]  ,
                answer_entity=data['answer_entity_text'],
                answer_triplet_index=data['answer_triplet_index'],
                sub_query=[data['question']],
                graph_id = data['question_id'],
            )

            yield kg  # 作为迭代器返回
