import re
import sys
import json

def extract_entities_from_sparql(sparql: str) -> set:
    # 匹配形如 ns:m.03wws_ 的 Freebase 实体
    pattern = r'\bns:(m\.[A-Za-z0-9_]+)\b'
    mids = re.findall(pattern, sparql)
    # 去重并保持顺序
    result = set()
    for m in mids:
        result.add(m)
    return result

if __name__ == "__main__":
    # 从 stdin 读取整个 SPARQL（你也可以改成读文件或直接写字符串）
    sparql = input()
    entities = extract_entities_from_sparql(sparql)
    # 输出期望格式：["m.03wws_"]
    print(entities)
