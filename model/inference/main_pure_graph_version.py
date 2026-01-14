from typing import Iterable, List, Sequence, Literal

import torch
from loguru import logger

from config.singleton import (
    CONV,
    DEFAULT_CHAT_MODEL,
    QTR_BODY,
    SCORER,
)
from utils.async_runner import AsyncRunner
from model.message_passing.utils import fetch_one_hop_triplets
from model.inference.prompt import (get_answer_prompt,
                                    potential_triplet_prompt, get_final_answer_prompt,
                                    find_useful_relations_prompt)
from model.message_passing.message_passing_subgraph_provider import (
    MessagePassingSubgraphProvider,
)
from model.message_passing.utils import update_qtr_emb
from model.schemas import KGDataset
from model.vocab.freebase_vocab import FreebaseVocab


def get_add_triplets(
    full_triples: Sequence[Sequence[str]],
    endpoints: Iterable[str],
    excluded_entities: Iterable[str] = None,
) -> List[List[str]]:
    """
    过滤三元组：
    - 保留主语或宾语出现在 endpoints 中的三元组
    - 且主语与宾语都不在 banned_entities 中
    - 仅对实体（主语、宾语）做禁用检查；谓词不参与禁用判断

    参数：
        triples: 序列，每个元素为长度为3的可迭代 [subject, predicate, object]
        endpoints: 需要匹配的实体ID集合（主语或宾语命中其一即可）
        excluded_entities: 不希望出现的实体ID集合（主语/宾语任一命中则剔除）

    返回：
        满足条件的三元组列表（每个元素为 [s, p, o]）
    """
    ep = set(endpoints)
    banned = set(excluded_entities) if excluded_entities is not None else set()
    out: List[List[str]] = []

    for t in full_triples:
        if len(t) != 3:
            continue  # 跳过非三元数据
        s, p, o = t[0], t[1], t[2]

        # 条件1：主语或宾语命中 endpoints
        if not (s in ep or o in ep):
            continue

        # 条件2：主语与宾语都不在 banned 中
        if (s in banned) or (o in banned):
            continue

        out.append([s, p, o])

    return out

class GraphReasoning:
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qtr_body = QTR_BODY.to(device)
    conv = CONV.to(device)
    scorer = SCORER.to(device)
    chat_model = DEFAULT_CHAT_MODEL


    triplet_text = []

    # # 解包kg
    # full_graph = []
    # query ="what was tupac name in juice?"
    # source_entity = ["m.07pzc"]
    # answer_entity = ["m.08w51z"]

    # 两类mask
    quality_mask = torch.tensor([0 for _ in range(len(triplet_text))], dtype=torch.bool)
    feasible_mask = torch.tensor([1 for _ in range(len(triplet_text))], dtype=torch.bool)
    # 用来标记大模型是否见过这个三元组，1表示没看过
    unseen_mask = torch.tensor([1 for _ in range(len(triplet_text))], dtype=torch.bool)

    # 记忆
    memory_triplet = []

    cached_triplet = []  # 用于存放超过长度上限的三元组，会在下一轮提供

    freebase_vocab = FreebaseVocab()

    # 提示词
    potential_triplet_prompt_template = potential_triplet_prompt
    get_answer_prompt_template = get_answer_prompt
    # get_answer_prompt_template = get_answer_prompt_with_entity_response
    get_final_answer_prompt_template = get_final_answer_prompt

    # 结束标记（搜到答案再搜一轮）
    answer_flag = 0


    for model in [qtr_body, conv, scorer]:
        for param in model.parameters():
            param.requires_grad = False

    def __init__(self, full_graph, query, source_entity, quality_threshold, feasible_threshold, expend_hop_num):
        self.full_graph=full_graph
        self.query = query
        self.source_entity = source_entity
        self.viewed_entity = set(source_entity.copy())
        # 超参数
        self.quality_threshold = quality_threshold
        self.feasible_threshold = feasible_threshold
        self.expend_hop_num = expend_hop_num

    async def just_one_conv_in_full_graph(self):
        # 图推理
        triplet_text = self.full_graph
        new_triplet_text = [t for t in triplet_text if t not in self.triplet_text]
        self.triplet_text = self.triplet_text + new_triplet_text
        new_l = len(new_triplet_text)

        device = self.quality_mask.device

        self.quality_mask = torch.cat([
            self.quality_mask,
            torch.zeros(new_l, dtype=torch.bool, device=device)
        ])
        self.feasible_mask = torch.cat([
            self.feasible_mask,
            torch.ones(new_l, dtype=torch.bool, device=device)
        ])
        self.unseen_mask = torch.cat([
            self.unseen_mask,
            torch.ones(new_l, dtype=torch.bool, device=device)
        ])

        # 可行子图 -> 全局子图 的index映射
        index_mapper = torch.tensor(
            [i for i, b in enumerate(self.unseen_mask & self.feasible_mask) if b],
            dtype=torch.long,
            device=device,
        )
        hop_triplets = [self.triplet_text[i] for i in index_mapper]

        kg = KGDataset(
            device=self.device,
            triplet_text=hop_triplets,
            query=self.query,
            source_entity=self.source_entity,
            # answer_entity=self.answer_entity,
        )
        await update_qtr_emb(kg)  # T,

        subgraph_provider = MessagePassingSubgraphProvider(
            kg=kg, directed=False, expand_threshold=self.feasible_threshold, quality_threshold=self.quality_threshold,
        )

        # 计算flow_emb_new
        flow = subgraph_provider.collect_within(self.expend_hop_num)
        triplet_idx = flow.triplet_index  # 子图里的 index
        src = flow.from_entity_index
        dst = flow.to_entity_index
        flow_emb = kg.message_flow_emb[src]  # [E_sub, h]
        qtr_emb = kg.qtr_emb[triplet_idx]
        flow_emb_new = self.conv(kg.query_emb, qtr_emb, flow_emb, src, dst, kg.num_entities)
        # kg.message_flow_emb = flow_emb_new

        # 收集两种score  写到这儿
        qtr_score = kg.qtr_score[triplet_idx]
        source_flow_emb = flow_emb_new[src]
        conv_logit = SCORER(qtr_emb, source_flow_emb)
        conv_score = torch.sigmoid(conv_logit).reshape(-1)

        edge_score = torch.max(qtr_score, conv_score)

        # 两类掩码（保持和全局 mask 同 device）
        step_quality_mask = torch.tensor(
            [i > self.quality_threshold for i in edge_score],
            dtype=torch.bool,
            device=device,
        )
        step_feasible_mask = torch.tensor(
            [i > self.feasible_threshold for i in edge_score],
            dtype=torch.bool,
            device=device,
        )

        # 子图索引 -> 全局索引
        global_triplet_idx = index_mapper[triplet_idx]

        self.feasible_mask[global_triplet_idx[~step_feasible_mask]] = False
        self.quality_mask[global_triplet_idx[step_quality_mask]] = True

        # 准备一跳领域的数据
        one_hop_flow = subgraph_provider.collect_between(1, 0)
        one_hop_mask = torch.zeros(
            len(self.triplet_text),
            dtype=torch.bool,
            device=device,
        )
        # one_hop_flow.triplet_index 是子图 index，先映射回全局
        one_hop_global_idx = index_mapper[one_hop_flow.triplet_index]
        one_hop_mask[one_hop_global_idx] = True


        # 大模型
        # 准备数据
        # 控制relation
        one_hop_mask = one_hop_mask & self.unseen_mask
        triplets_to_llm = [self.triplet_text[i] for i, b in enumerate(one_hop_mask) if b]
        relations = list(set([t[1] for t in triplets_to_llm]))
        prompt = find_useful_relations_prompt.format(relations=relations, query=self.query)
        llm_response = await self.chat_model.chat_with_json_response(
            prompt,
            retried_times=2,
        )
        one_hop_input_triplets = [t for t in triplets_to_llm if t[1] in llm_response['useful_relations']]


        # quality
        llm_mask = (self.quality_mask) & self.unseen_mask
        quality_triplets = [self.triplet_text[i] for i, b in enumerate(llm_mask) if b]
        self.unseen_mask[llm_mask | one_hop_mask] = False
        llm_input_triplets = list(self.memory_triplet) + quality_triplets + one_hop_input_triplets
        llm_input_triplets = [list(t) for t in dict.fromkeys(map(tuple, llm_input_triplets))]  # 去重


        # 控制输入上限
        llm_input_triplets = self.cached_triplet + llm_input_triplets
        if len(llm_input_triplets) > 1000:
            self.cached_triplet = llm_input_triplets[1000:]

        # 获取潜在子图
        # step_memory_triplet = [self.triplet_text[i] for i, m in enumerate(self.feasible_mask) if m]
        get_potential_triplet_prompt = self.potential_triplet_prompt_template.format(
            query=self.query,
            triplets=llm_input_triplets,
        )
        potential_triplet = await self.chat_model.chat_with_json_response(
            get_potential_triplet_prompt,
            retried_times=2,
        )

        # 尝试解码未知id
        decode_lst = []
        for s, r, o in potential_triplet:
            if s.startswith("m.") or s.startswith("g."):
                decode_lst.append(s)
            if o.startswith("m.") or o.startswith("g."):
                decode_lst.append(o)

        neighbor_triplets_text = []
        if decode_lst:
            neighbor_triplets_id = await AsyncRunner(
                fetch_one_hop_triplets,
                max_concurrency=10,
            ).run(decode_lst)
            neighbor_triplets_id = [t for i in neighbor_triplets_id for t in i]
            neighbor_triplets_text = await self.freebase_vocab.decode_triple(
                neighbor_triplets_id
            )

        # 这里 potential_triplet 已经包含“原始候选 + 解码后的邻居三元组”
        potential_triplet += neighbor_triplets_text



        # —— 关键点：把 decode 出来的邻居三元组同步到 full_graph / triplet_text / 各种 mask —— #
        if neighbor_triplets_text:
            # 先和 full_graph 去重，再写回
            full_set = set(map(tuple, self.full_graph))
            new_from_neighbor = [t for t in neighbor_triplets_text if tuple(t) not in full_set]
            if new_from_neighbor:
                self.full_graph += new_from_neighbor

                # 再保证 self.triplet_text / 各种 mask 也同步
                triplet_set = set(map(tuple, self.triplet_text))
                append_to_triplet = [t for t in new_from_neighbor if tuple(t) not in triplet_set]
                extra_l = len(append_to_triplet)
                if extra_l > 0:
                    self.triplet_text += append_to_triplet
                    self.quality_mask = torch.cat([
                        self.quality_mask,
                        torch.zeros(extra_l, dtype=torch.bool, device=device),
                    ])
                    self.feasible_mask = torch.cat([
                        self.feasible_mask,
                        torch.ones(extra_l, dtype=torch.bool, device=device),
                    ])
                    self.unseen_mask = torch.cat([
                        self.unseen_mask,
                        torch.ones(extra_l, dtype=torch.bool, device=device),
                    ])

        # 获取答案
        answer_prompt = self.get_answer_prompt_template.format(
            query=self.query,
            triplets=list(self.memory_triplet) + potential_triplet,
        )
        llm_res = await self.chat_model.chat_with_json_response(
            answer_prompt,
            retried_times=2,
        )

        # 检查下一跳实体是否在候选里
        check_e = [[u, v] for u, r, v in (list(self.memory_triplet) + potential_triplet)]
        check_e = [i for j in check_e for i in j]
        next_hop_source_entity = [e for e in check_e if e not in self.viewed_entity]

        # 下一条的起始实体
        self.viewed_entity |= set(next_hop_source_entity)
        self.source_entity = next_hop_source_entity

        return {**llm_res, "potential_triplet": potential_triplet}


    async def inference_in_full_graph(self):
        while True:
            if len(self.source_entity) != 0:
                llm_response = await self.just_one_conv_in_full_graph()
                print(llm_response)
                if llm_response["predict_answer"] != []:
                    if self.answer_flag != 1:
                        self.answer_flag = 1
                    else:
                        return llm_response
            else:
                logger.warning("搜索到尽头无确定答案")
                logger.warning(f"{self.memory_triplet}")
                answer_prompt = self.get_final_answer_prompt_template.format(query=self.query, triplets=list(
                    self.memory_triplet) + self.memory_triplet)
                llm_res = await self.chat_model.chat_with_json_response(answer_prompt, retried_times=2)
                return llm_res

            # 更新记忆
            self.memory_triplet = [list(k) for k in dict.fromkeys(map(tuple, self.memory_triplet + llm_response["potential_triplet"]))]


async def main(ds_type: Literal["webqsp", "cwq"], save_jsonl_path: str, ques_index: List[int] = None,
               max_concurrency=5,
               feasible_threshold=0, quality_threshold=0.7):
    from datasets import load_dataset
    from utils.async_jsonl_writer import AsyncJsonlWriter
    from utils.async_runner import AsyncRunner
    import json

    triplet_dict = {}

    if ds_type == "webqsp":
        ds = load_dataset("Youm9602/RoG-webqsp", split="test")
        with open('./files/webqsp/test_triplet.jsonl') as f:
            for line in f:
                data = json.loads(line)
                triplet_dict[data["QuestionId"]] = data
    elif ds_type == "cwq":
        ds = load_dataset("rmanluo/RoG-cwq", split="test")
        with open('./files/cwq/test_triplet.jsonl') as f:
            for line in f:
                data = json.loads(line)
                triplet_dict[data["QuestionId"]] = data
        pass
    else:
        raise ValueError(f"dataset {ds_type} not supported")


    async def single_task(task, writer, **kwargs):
        # 解包kg
        id = task["id"]
        full_graph = task["graph"]
        query = task["question"]
        source_entity = task["q_entity"]
        answer_entity = task["a_entity"]

        full_graph = triplet_dict[id]["gold_triplet_text_lst"] + full_graph
        full_graph = [list(t) for t in dict.fromkeys(map(tuple, full_graph))]
        source_entity.extend(triplet_dict[id]["gold_source_entity_text_lst"])
        source_entity = list(set(source_entity))

        reasoner = GraphReasoning(full_graph=full_graph, query=query, source_entity=source_entity,
                                  feasible_threshold=feasible_threshold, quality_threshold=quality_threshold, expend_hop_num=2)
        llm_response = await reasoner.inference_in_full_graph()
        predict_answer = llm_response.get("predict_answer")

        save_json = {**dict(task), **llm_response, **kwargs}
        await writer.write(save_json)

        del reasoner

        return predict_answer


    async with AsyncJsonlWriter(save_jsonl_path) as writer:
        runner = AsyncRunner(single_task, max_concurrency= max_concurrency, writer=writer)
        tasks = []
        ques_index = [i for i in range(100)] if ques_index is None else ques_index
        for i in ques_index:
            tasks.append(runner.append(ds[i], ques_index=i))
        res = [await task for task in tasks]

    return res





if __name__ == '__main__':
    import asyncio
    asyncio.run(main())