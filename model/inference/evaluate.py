import json
from typing import Literal

def main(ds_type=Literal["cwq", "webqsp"]):
    data = []

    if ds_type == "cwq":
        file_path = "./files/result/cwq/test.jsonl"
    elif ds_type == "webqsp":
        file_path = "./files/result/webqsp/test.jsonl"
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))


    # 计算hit@1，hit@all，precision，recall，f1
    hit_1 = []
    hit_all = []
    precision = []
    recall = []
    f1_score = []

    for line in data:
        ques_id = line["id"]
        ques_index = line["ques_index"]
        predict_answer = [p.lower() for p in line['predict_answer']]
        gold_answer = [a.lower() for a in line['answer']]
        source_entity = line['q_entity']

        if predict_answer == []:
            print(1)
            # e = [[t[0], t[2]] for t in line['potential_triplet']]
            # e = [i for o in e for i in o]
            # e = [i for i in e if not i.startswith("m.")]
            # e = [i for i in e if not i in source_entity]

        if predict_answer == []:
            pass

        # 根据predict_answer和gold_answer计算hit@1，hit@all，precision，recall，f1_score
        # 计算correct
        correct = 0
        for p in predict_answer:
            if p in gold_answer:
                correct += 1
        total = len(gold_answer)
        if total > 0:
            if predict_answer != []:
                hit_1.append(1 if predict_answer[0] in gold_answer else 0)
                precision.append(correct / len(predict_answer))
            hit_all.append(1 if correct > 0 else 0)
            recall.append(correct / total)
            if precision[-1] + recall[-1] == 0:
                f1_score.append(0)
            else:
                f1_score.append(2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1]))

        # for p in predict_answer:
        #     if p in gold_answer:
        #         correct += 1
        #         break
        # for p in predict_answer[:1]:
        #     if p in gold_answer:
        #         correct += 1
        #         break

        if hit_1[-1] == 0:
            print(f"{ques_id = }")
            print(f"{ques_index = }")
            print(f"{line['predict_answer'] = }")
            print(f"{line['answer'] = }")
            # print(f"{line['potential_triplet'] = }")
            print("="*100)


    # print 各个指标
    print(f"hit@1: {sum(hit_1) / len(hit_1)}")
    print(f"hit@all: {sum(hit_all) / len(hit_all)}")
    print(f"precision: {sum(precision) / len(precision)}")
    print(f"recall: {sum(recall) / len(recall)}")
    print(f"f1_score: {sum(f1_score) / len(f1_score)}")

    print(recall)