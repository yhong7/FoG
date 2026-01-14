from model.build_freebase_dataset.sparql_to_construct import sparql_to_construct
from model.vocab.freebase_vocab import FreebaseVocab
import json
from model.utils import extract_entities_from_sparql

freebase_vocab = FreebaseVocab()


async def get_gold_triplets(q, writer, dataset_name:str):
    sparql_set = set()
    construct_sparql_set = set()
    source_entity_id_lst = []
    source_entity_text_lst = []
    gold_triplet_lst = []
    gold_answer_set = set()

    if dataset_name == "webqsp":
        for parse in q["Parses"]:
            sparql_set.add(parse["Sparql"])
            source_entity_id_lst.append(parse["TopicEntityMid"])
            source_entity_text_lst.append(parse["TopicEntityName"])
            # source_entity_dict[parse["TopicEntityMid"]] = parse["TopicEntityName"]
            for a in parse["Answers"]:
                gold_answer_set.add(a["EntityName"])
        question_id = q["QuestionId"]
        question = q["ProcessedQuestion"]

    elif dataset_name == "cwq":
        sparql_set.add(q["sparql"])
        question_id = q["ID"]
        question = q["question"]
        sparql_res = await freebase_vocab.sparql(q["sparql"])
        gold_ids = [i["x"]["value"] for i in sparql_res["results"]["bindings"]]
        gold_ids = [url_id.split("freebase.com/ns/")[1] for url_id in gold_ids]
        for gold_id in gold_ids:
            gold_answer_set.add(await freebase_vocab.decode_entity(gold_id))

    else:
        raise ValueError("dataset_name must be either 'webqsp' or 'cwq'")


    for sparql in sparql_set:
        try:
            construct_sparql = sparql_to_construct(sparql)
            construct_sparql_set.add(construct_sparql)
            gold_id_triplets = await freebase_vocab.sparql_with_triplet_res(construct_sparql)
            gold_triplet_lst.extend(gold_id_triplets)
        except Exception as e:
            print(e)

    # 去重：
    gold_triplet_lst = [list(t) for t in dict.fromkeys(map(tuple, gold_triplet_lst))]
    gold_triplet_text_lst = await freebase_vocab.decode_triple(gold_triplet_lst)

    # 抽起始实体
    sparql_lst = list(sparql_set)
    sparql_e_id = [extract_entities_from_sparql(q) for q in sparql_lst]
    sparql_e_id = [i for j in sparql_e_id for i in j]
    sparql_text = await freebase_vocab.decode_entity(sparql_e_id)
    source_entity_id_lst.extend(sparql_e_id)
    source_entity_text_lst.extend(sparql_text)
    source_entity_id_lst = list(set(source_entity_id_lst))
    source_entity_text_lst = list(set(source_entity_text_lst))

    ret_dic = {
        "QuestionId": question_id,
        "ProcessedQuestion": question,
        "sparql_lst": sparql_lst,
        "gold_triplet_text_lst": gold_triplet_text_lst,
        "gold_source_entity_id_lst": source_entity_id_lst,
        "gold_source_entity_text_lst": source_entity_text_lst,
        "gold_answer_set": list(gold_answer_set),
    }
    await writer.write(ret_dic)

async def main(ds_type: str, save_path: str, max_concurrency=5):
    from utils.async_runner import AsyncRunner
    from utils.async_jsonl_writer import AsyncJsonlWriter

    if ds_type.lower() == "webqsp":
        ori_dataset_path = "./files/WebQSP_official/data/WebQSP.test.json"

        with open(ori_dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)["Questions"]

            async with AsyncJsonlWriter(save_path) as writer:
                runner = AsyncRunner(get_gold_triplets, writer=writer, max_concurrency=10, dataset_name=ds_type)
                await runner.run(dataset)

    elif ds_type.lower() == "cwq":
        from datasets import load_dataset
        dataset = load_dataset("drt/complex_web_questions", "complexwebquestions_test", split="test")

        async with AsyncJsonlWriter(save_path) as writer:
            runner = AsyncRunner(get_gold_triplets, writer=writer, max_concurrency=max_concurrency, dataset_name=ds_type)
            tasks = []
            for line in dataset:
                tasks.append(runner.append(line))
            [await task for task in tasks]



if __name__ == '__main__':
    pass


