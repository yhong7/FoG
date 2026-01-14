import random

from model.build_freebase_dataset.sparql_to_construct import sparql_to_construct
import json
from model.vocab.freebase_vocab import FreebaseVocab
from model.build_freebase_dataset.expand_freebase_trainset import expand_freebase_parallel
from utils.async_jsonl_writer import AsyncJsonlWriter
from utils.async_runner import AsyncRunner
from loguru import logger
import asyncio
from typing import Literal

logger.add("files/CWQ/train.log", )
random.seed(42)

async def process_single_row(q, vocab, writer, ds_type: str):
    if ds_type.lower() == "webqsp":
        question_id = q["QuestionId"]
        question = q["ProcessedQuestion"]
        sparql = q["Parses"][0]["Sparql"]
        answers = q["Parses"][0]["Answers"]
    elif ds_type.lower() == "cwq":
        question_id = q["webqsp_ID"]
        question = q["question"]
        sparql = q["sparql"]
        answers = q["answers"]["answer_id"]
    else:
        raise ValueError(f"Unsupported dataset type: {ds_type}")

    try:
        construct_sparql = sparql_to_construct(sparql)
        # gold子图
        gold_id_triplets = await vocab.sparql_with_triplet_res(construct_sparql)
        # gold_triplets = await vocab.decode_triple(gold_id_triplets)

        # 负样本
        negative_id_triplets = await expand_freebase_parallel(gold_id_triplets, hops=2, limit_per_hop=3)
        # negative_triplets = await vocab.decode_triple(negative_id_triplets)

        # concat
        full_train_id_set = gold_id_triplets + negative_id_triplets
        full_train_set = await vocab.decode_triple(full_train_id_set)
        gold_triplet_idx = [i for i in range(len(gold_id_triplets))]
        print(len(gold_id_triplets), len(full_train_set))

        # 部分数据存在问题
        if len(negative_id_triplets) == 0:
            return
        if len(gold_id_triplets) > 50:
            return

        # 起始实体
        q_entity = gold_id_triplets[0][0]
        q_entity_text = await vocab.decode_entity(q_entity)

        # 答案实体
        if ds_type == "webqsp":
            a_entity = [a['AnswerArgument'] for a in answers]
            a_entity_text = await vocab.decode_entity(a_entity)
        elif ds_type == "cwq":
            a_entity = q["answers"]["answer_id"]
            a_entity_text = q["answers"]["answer"]

    except Exception as e:
        print(f"exception: {e}")
        print(f"sparql: {sparql}")
        return

    await writer.write({
        "question_id": question_id,
        "question": question,
        "sparql": sparql,
        "construct_sparql": construct_sparql,
        "source_entity": q_entity,
        "source_entity_text": q_entity_text,
        "answer_entity": a_entity,
        "answer_entity_text": a_entity_text,
        "gold_id_triplets": gold_id_triplets,
        "negative_id_triplets": negative_id_triplets,
        "full_id_triplets": full_train_id_set,
        "full_text_triplet": full_train_set,
        "answer_triplet_index": gold_triplet_idx,
    })

    logger.success(f"ID: {question_id} Success")


async def main(save_path: str, ds_type: Literal["webqsp", "cwq"], max_concurrency=10):
    async with AsyncJsonlWriter(save_path) as writer:
        vocab = FreebaseVocab()
        runner = AsyncRunner(process_single_row, max_concurrency=max_concurrency, vocab=vocab, writer=writer, ds_type=ds_type)

        if ds_type.lower() == "webqsp":
            file_path = "./files/WebQSP_official/data/WebQSP.train.json"
            with open(file_path, "r", encoding="utf-8") as f:
                file = json.load(f)
            await runner.run(file["Questions"])

        elif ds_type.lower() == "cwq":
            from datasets import load_dataset
            ds = load_dataset("drt/complex_web_questions", "complex_web_questions", trust_remote_code=True, split="train")
            await runner.run(ds)

        # for q in file["Questions"]:


if __name__ == "__main__":
    import asyncio
    # save_path = "./files/WebQSP_official/processed.jsonl"
    save_path = "./files/CWQ/train.jsonl"
    asyncio.run(main(save_path, ds_type="cwq"))
    # asyncio.run(temp())