import re
from textwrap import indent
import re
from textwrap import indent

def _prep_for_construct_extraction(where_body: str) -> str:
    # 仅用于“抽取副本”：去注释 -> 去 FILTER(...) -> 去 EXISTS/NOT EXISTS/SERVICE {...}
    s = "\n".join(_strip_line_comment(ln) for ln in where_body.splitlines())
    s = _remove_all_FILTER_paren_blocks(s)
    s = _remove_keyword_brace_blocks(s, keywords=('not exists', 'exists', 'service'))
    return s

def _expand_semicolon_block(block: str) -> list[str]:
    """
    将形如：
      s p1 o1 ; p2 o2 ; p3 o3 .
    的语句块展开成多条三元组：
      s p1 o1 .
      s p2 o2 .
      s p3 o3 .
    仅处理 ';' 缩写，不额外增加其它功能。
    """
    block = block.strip()
    if not block.endswith('.'):
        return []

    # 去掉末尾的 '.'
    body = block[:-1].strip()
    # 归一化空白
    body = re.sub(r'\s+', ' ', body)

    # 如果本身没有 ';'，直接返回原始块（交给外层处理）
    if ';' not in body:
        return [block]

    # 按 ';' 切分：第一个片段包含 subject
    segments = [seg.strip() for seg in body.split(';') if seg.strip()]
    if not segments:
        return []

    first_seg = segments[0]
    first_tokens = first_seg.split(' ')
    # 至少要有 subject、predicate、object 三部分，不然不展开
    if len(first_tokens) < 3:
        return [block]

    subject = first_tokens[0]
    triples: list[str] = []

    # 第一条：原片段 + '.'
    triples.append(f"{first_seg} .")

    # 后续片段：补上 subject
    for seg in segments[1:]:
        if not seg:
            continue
        triples.append(f"{subject} {seg} .")

    return triples


def _strip_line_comment(s: str) -> str:
    return s.split('#', 1)[0]

def _scan_balanced(s: str, i: int, open_ch: str, close_ch: str) -> int:
    """
    从 s[i] 开始，假设 s[i] == open_ch，向右扫描到与之配对的 close_ch 的索引（含）。
    若不配对，返回 -1。
    """
    if i >= len(s) or s[i] != open_ch:
        return -1
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1

def _remove_all_FILTER_paren_blocks(s: str) -> str:
    """
    线性扫描，删除所有形如  FILTER ( ...配对... )  的片段；大小写不敏感；允许嵌套括号。
    仅用于“抽取构造三元组”的副本；原 WHERE 保留。
    """
    out = []
    i = 0
    L = len(s)
    lower = s.lower()
    while i < L:
        # 找到 "filter" 作为单词边界
        if lower.startswith('filter', i) and (i == 0 or not lower[i-1].isalnum()) \
           and (i+6 >= L or not lower[i+6].isalnum()):
            j = i + 6
            # 跳过空白
            while j < L and s[j].isspace():
                j += 1
            # 必须紧跟 '('
            if j < L and s[j] == '(':
                end = _scan_balanced(s, j, '(', ')')
                if end != -1:
                    # 丢弃 [i : end+1] 这段
                    i = end + 1
                    continue
            # 如果不像 FILTER(...)，就当普通文本输出
        out.append(s[i])
        i += 1
    return ''.join(out)

def _remove_keyword_brace_blocks(s: str, keywords=('exists', 'not exists', 'service')) -> str:
    """
    删除  KEYWORD { ...配对... }  的片段，大小写不敏感；允许嵌套花括号。
    仅用于“抽取构造三元组”的副本。
    """
    out = []
    i = 0
    L = len(s)
    lower = s.lower()
    kw_list = sorted(keywords, key=len, reverse=True)  # 先长后短，避免前缀误匹配
    while i < L:
        matched = False
        for kw in kw_list:
            k = len(kw)
            if lower.startswith(kw, i) and (i == 0 or not lower[i-1].isalnum()) \
               and (i+k >= L or not lower[i+k].isalnum()):
                j = i + k
                while j < L and s[j].isspace():
                    j += 1
                if j < L and s[j] == '{':
                    end = _scan_balanced(s, j, '{', '}')
                    if end != -1:
                        i = end + 1   # 丢掉这个块
                        matched = True
                        break
        if matched:
            continue
        out.append(s[i])
        i += 1
    return ''.join(out)

def _find_where_body(q: str) -> str:
    m = re.search(r'\bWHERE\b', q, flags=re.IGNORECASE)
    if not m:
        raise ValueError("没有找到 WHERE 关键字。")
    i = q.find('{', m.end())
    if i == -1:
        raise ValueError("WHERE 后没有找到 '{' 。")
    end = _scan_balanced(q, i, '{', '}')
    if end == -1:
        raise ValueError("没有找到 WHERE { ... } 的配对 '}' 。")
    return q[i+1:end].strip()

def _extract_construct_triples(where_body: str) -> list[str]:
    s = _prep_for_construct_extraction(where_body)

    # 先按“语句块”而不是“单行”来聚合：
    # 连续的多行，直到遇到以 '.' 结尾的行，合成为一个 block
    blocks: list[str] = []
    current_lines: list[str] = []

    for raw_line in s.split('\n'):
        line = _strip_line_comment(raw_line).strip()
        if not line:
            continue
        # 这些行不用于 CONSTRUCT 模板
        if re.match(r'^(FILTER|BIND|VALUES|SERVICE)\b', line, flags=re.IGNORECASE):
            continue

        # OPTIONAL 去掉关键字本身
        line = re.sub(r'\bOPTIONAL\b', ' ', line, flags=re.IGNORECASE)
        # 花括号直接抹掉，只保留里面的三元组
        line = line.replace('{', ' ').replace('}', ' ').strip()
        if not line:
            continue

        # 关键：UNION 不应该进入 CONSTRUCT 模板，直接跳过这一行
        if re.match(r'^UNION\b', line, flags=re.IGNORECASE):
            continue

        # 累积当前语句块
        current_lines.append(line)

        # 以 '.' 结尾代表一个语句结束
        if line.endswith('.'):
            block = ' '.join(current_lines).strip()
            blocks.append(block)
            current_lines = []

    triples: list[str] = []
    seen: set[str] = set()

    for block in blocks:
        # 归一化空白
        block = re.sub(r'\s+', ' ', block).strip()
        if not block.endswith('.'):
            continue

        body = block[:-1].strip()
        # 只剩 '.' 或空，直接丢弃，避免生成孤立的 '.' 三元组
        if not body:
            continue

        # 包含 ';' 的，做缩写展开
        if ';' in block:
            expanded = _expand_semicolon_block(block)
            for t in expanded:
                t = re.sub(r'\s+', ' ', t).strip()
                if not t.endswith('.'):
                    continue
                b2 = t[:-1].strip()
                # 至少要有 subject、predicate、object 三部分
                if len(b2.split()) < 3:
                    continue
                if t not in seen:
                    seen.add(t)
                    triples.append(t)
        else:
            # 普通三元组：至少要有 3 个 token
            if len(body.split()) < 3:
                continue
            if block not in seen:
                seen.add(block)
                triples.append(block)

    if not triples:
        raise ValueError("未能在 WHERE 中识别到三元组模式。")
    return triples

def sparql_to_construct(query: str) -> str:
    """
    将常见 SELECT SPARQL 转成返回匹配子图的 CONSTRUCT 查询。
    - 保留 PREFIX/BASE 与 WHERE 中的 FILTER/OPTIONAL 等
    - 将 WHERE 中出现的三元组模式复制到 CONSTRUCT 中（忽略 EXISTS/NOT EXISTS/SERVICE 块内的三元组）
    """
    # 1) 归一化换行与空白
    q = query.replace("\\n", "\n")
    # 不在这里全局压缩空白，避免破坏字符串字面量与日期字面量
    q = re.sub(r'[ \t]+', ' ', q).strip()

    # 2) 收集 PREFIX/BASE（大小写不敏感）
    prefix_lines = re.findall(r'^(?:PREFIX|BASE)\s+.+$', q, flags=re.MULTILINE | re.IGNORECASE)
    prefix_block = "\n".join(prefix_lines).strip()

    # 3) 精确截取 WHERE 正文（允许后面有 ORDER BY / LIMIT 等）
    where_body = _find_where_body(q)

    # 4) 从 WHERE 中抽取将用于 CONSTRUCT 的三元组
    construct_lines = _extract_construct_triples(where_body)

    # 5) 组装最终查询
    final = []
    if prefix_block:
        final.append(prefix_block)
    final.append("CONSTRUCT {")
    final.append(indent("\n".join(construct_lines), "  "))
    final.append("}")
    final.append("WHERE {")
    final.append(indent(where_body, "  "))
    final.append("}")
    return "\n".join(final)


# ---- 演示：用题目中的 SPARQL ----
if __name__ == "__main__":
    input_sparql = """
    PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.06w2sn5)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.06w2sn5 ns:people.person.sibling_s ?y .\n?y ns:people.sibling_relationship.sibling ?x .\n?x ns:people.person.gender ns:m.05zppz .\n}\n"""
    print(sparql_to_construct(input_sparql))
