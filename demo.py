import os
import re
import ast
import json
import time
import copy
import streamlit as st

st.set_page_config(page_title="GRPO-Lawyer Demo", layout="wide")
# Set up hyperparams
DEMO_FILE = "demo.jsonl"
TRANSLATED_FILE = "translated.jsonl"
AVATARS = {"assistant": "🧑‍⚖️", "user": "🧑‍💼", "system": "⚙️"}

SRC_LANG = "zh"   
TGT_LANG = "en"

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
DB_PREFIX = "[Database Result]"


"""
Helper translation function using argotranslate for offline efficient translations.

Downloads and installs the language package on first use (needs internet once).
Gets Argos `Translation` object for zh→en.

Returns: (translation, error_message).
"""
@st.cache_resource(show_spinner="Loading Argos Translate zh→en model…")
def get_argos_translation():

    try:
        import argostranslate.package
        import argostranslate.translate
    except ImportError:
        return None, "Please run: `pip install argostranslate`"

    def find_installed():
        langs = argostranslate.translate.get_installed_languages()
        src = next((l for l in langs if l.code == SRC_LANG), None)
        tgt = next((l for l in langs if l.code == TGT_LANG), None)
        if src is None or tgt is None:
            return None
        return src.get_translation(tgt)

    tr = find_installed()
    if tr is not None:
        return tr, None

    try:
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        pkg = next(
            (p for p in available if p.from_code == SRC_LANG and p.to_code == TGT_LANG),
            None,
        )
        if pkg is None:
            return None, f"No Argos package found for {SRC_LANG}→{TGT_LANG}."
        argostranslate.package.install_from_path(pkg.download())
    except Exception as e:
        return None, (
            f"Could not download the Argos {SRC_LANG}→{TGT_LANG} model "
            f"({type(e).__name__}: {e}). Install it manually with "
            f"`argospm install translate-{SRC_LANG}_{TGT_LANG}` and restart."
        )

    tr = find_installed()
    if tr is None:
        return None, "Model installed but could not be loaded. Restart the app."
    return tr, None


"""
Translation function that translates chinese text to english using argotranslate

Translate a block of text while preserving line structure.
Lines with no CJK characters (numbers, English, markdown rules…) are passed through untouched.

Returns - Translated text
"""
def translate_text(text, translation, stats):

    if not isinstance(text, str) or not text.strip():
        return text

    out_lines = []
    for line in text.split("\n"):
        if not CJK_RE.search(line):
            out_lines.append(line)
            continue

        # Preserve leading bullet / numbering so markdown lists survive
        m = re.match(r"^(\s*(?:[-*•]|\d+[.)、]|[（(]\d+[)）])\s*)(.*)$", line)
        prefix, body = (m.group(1), m.group(2)) if m else ("", line)

        try:
            res = translation.translate(body.strip())
        except Exception as e:
            stats["errors"].append(f"{type(e).__name__}: {e}")
            res = None

        if res and res.strip():
            stats["ok"] += 1
            if CJK_RE.search(res):
                stats["residual_cjk"] += 1
            out_lines.append(prefix + res.strip())
        else:
            stats["failed"] += 1
            out_lines.append(line)

    return "\n".join(out_lines)


"""
Function that extracts a clean list of individual milestone strings.
Used to extract and translate the milestone strings from the json

Parameters:
raw - the raw text to be processed

Returns:
The list of individual milestone strings
"""
def parse_zh_milestones(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        flattened = []
        for item in raw:
            if isinstance(item, list):
                flattened.extend(parse_zh_milestones(item))
            elif isinstance(item, str):
                c = item.strip(" '\"[]\t\n")
                if c:
                    flattened.append(c)
        return flattened

    if not isinstance(raw, str) or not raw.strip():
        return []

    text = raw.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            data = loader(text)
            if isinstance(data, (list, tuple)):
                return [str(x).strip(" '\"[]") for x in data if str(x).strip(" '\"[]")]
        except Exception:
            pass

    if "\n" in text:
        return [l.strip(" '\"-•*[]") for l in text.split("\n") if l.strip(" '\"-•*[]")]

    return [text.strip(" '\"[]")]

"""
Function that translates the text blocks or milestone strings

Parameters:
content - the content to be translated
translation - argotranslate item
stats - statistics item

Returns:
content - the translated content
"""
def translate_message_content(content, translation, stats):
    if isinstance(content, str):
        if content.startswith(DB_PREFIX):
            _, _, body = content.partition(":")
            return f"{DB_PREFIX}: {translate_text(body.strip(), translation, stats)}"
        return translate_text(content, translation, stats)

    if isinstance(content, list):
        new_parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                p = dict(part)
                p["text"] = translate_text(p["text"], translation, stats)
                new_parts.append(p)
            elif isinstance(part, str):
                new_parts.append(translate_text(part, translation, stats))
            else:
                new_parts.append(part)
        return new_parts

    return content


"""
Function which generates the translated en jsonl file.

Returns:
True/False - Whether the translated file was successfully translated
"""
def generate_translated_file():
    if not os.path.exists(DEMO_FILE):
        st.error(f"Source file `{DEMO_FILE}` not found.")
        return False

    translation, err = get_argos_translation()
    if translation is None:
        st.error(err)
        return False

    with open(DEMO_FILE, "r", encoding="utf-8") as f:
        raw_lines = [l.strip() for l in f if l.strip()]
    total = len(raw_lines)
    if total == 0:
        st.error(f"`{DEMO_FILE}` is empty.")
        return False

    status_box = st.empty()
    progress_bar = st.progress(0.0)
    stats = {"ok": 0, "failed": 0, "residual_cjk": 0, "errors": []}
    translated_cases = []
    t0 = time.time()

    for idx, line in enumerate(raw_lines):
        try:
            c = copy.deepcopy(json.loads(line))
        except json.JSONDecodeError as e:
            st.error(f"Line {idx + 1} of `{DEMO_FILE}` is not valid JSON: {e}")
            return False

        status_box.info(
            f"🌐 Translating case {idx + 1}/{total} (ID: {c.get('case_id', idx)}) "
            f"— {time.time() - t0:.0f}s elapsed"
        )

        # Facts
        if isinstance(c.get("fact"), str):
            c["fact"] = translate_text(c["fact"], translation, stats)

        # Milestones (1:1 list)
        c["milestones"] = [
            translate_text(m, translation, stats)
            for m in parse_zh_milestones(c.get("milestones", []))
        ]

        # Messages
        if isinstance(c.get("messages"), list):
            for msg in c["messages"]:
                if "content" in msg and msg["content"]:
                    msg["content"] = translate_message_content(msg["content"], translation, stats)

        translated_cases.append(c)
        progress_bar.progress((idx + 1) / total)

    if stats["ok"] == 0 and stats["failed"] > 0:
        st.error(
            "Every segment failed to translate — not writing `translated.jsonl`. "
            f"Sample errors: {stats['errors'][:3]}"
        )
        progress_bar.empty()
        return False

    with open(TRANSLATED_FILE, "w", encoding="utf-8") as f:
        for item in translated_cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    load_file.clear()
    msg = (
        f"🎉 Translation complete in {time.time() - t0:.0f}s → `{TRANSLATED_FILE}` "
        f"({stats['ok']} segments translated"
    )
    if stats["failed"]:
        msg += f", {stats['failed']} failed and kept in Chinese"
    if stats["residual_cjk"]:
        msg += f", {stats['residual_cjk']} still contain some Chinese characters"
    msg += ")"
    status_box.success(msg)
    if stats["errors"]:
        st.warning(f"Errors during translation (first 3): {stats['errors'][:3]}")
    time.sleep(1.5)
    status_box.empty()
    progress_bar.empty()
    return True


@st.cache_data
def load_file(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def count_cjk_cases(cases):
    n = 0
    for c in cases:
        blob = json.dumps(c, ensure_ascii=False)
        if CJK_RE.search(blob):
            n += 1
    return n


# Data Loading
if not os.path.exists(TRANSLATED_FILE):
    generate_translated_file()

raw_zh_cases = load_file(DEMO_FILE)
raw_en_cases = load_file(TRANSLATED_FILE) if os.path.exists(TRANSLATED_FILE) else []

if not raw_zh_cases:
    st.error(f"Source file `{DEMO_FILE}` not found or empty.")
    st.stop()

if "selected_case_idx" not in st.session_state:
    st.session_state.selected_case_idx = 0
if st.session_state.selected_case_idx >= len(raw_zh_cases):
    st.session_state.selected_case_idx = 0


# Set up UI alternate texts for both languages
I18N = {
    "en": {
        "title": "GRPO-Lawyer",
        "select_case": "Select Case",
        "case_facts": "Case facts / 案情",
        "milestones": "Milestones Progress",
        "law_found_title": "Legal Basis Identified",
        "db_result": "🔍 Legal Database Result",
        "play": "▶ Play",
        "reveal": "Reveal Messages",
        "speed": "Playback Delay (sec/turn)",
        "m_milestones": "Milestones",
        "m_law_found": "Law found",
        "m_searches": "Searches",
        "m_reward": "Reward",
        "case_prefix": "Case #",
        "retranslate_btn": "🔄 Re-translate demo.jsonl (Argos)",
        "no_translation": "No `translated.jsonl` found — showing Chinese. Click Re-translate.",
        "count_mismatch": "`translated.jsonl` has {en} cases but `demo.jsonl` has {zh}. Showing Chinese — please Re-translate.",
        "residual": "{n} translated case(s) still contain Chinese characters.",
    },
    "zh": {
        "title": "GRPO-Lawyer",
        "select_case": "选择案例",
        "case_facts": "案情简述",
        "milestones": "里程碑完成进度",
        "law_found_title": "相关法条已检索命中",
        "db_result": "🔍 数据库检索结果",
        "play": "▶ 播放",
        "reveal": "展示消息进度",
        "speed": "播放间隔 (秒/轮)",
        "m_milestones": "里程碑占比",
        "m_law_found": "法条命中",
        "m_searches": "检索次数",
        "m_reward": "最终得分",
        "case_prefix": "案例 #",
        "retranslate_btn": "🔄 强制重新翻译文件 (Argos)",
        "no_translation": "未找到 translated.jsonl，显示中文。",
        "count_mismatch": "translated.jsonl 有 {en} 个案例，demo.jsonl 有 {zh} 个，显示中文。",
        "residual": "{n} 个翻译案例仍含中文字符。",
    },
}


# Set up UI using Streamlit
with st.sidebar:
    st.title("GRPO-Lawyer")

    lang_choice = st.radio("Language / 语言", ["English", "中文"], horizontal=True)
    lang_key = "en" if lang_choice == "English" else "zh"
    t = I18N[lang_key]

    if st.button(t["retranslate_btn"], use_container_width=True):
        if os.path.exists(TRANSLATED_FILE):
            os.remove(TRANSLATED_FILE)
        if generate_translated_file():
            st.rerun()

    if lang_key == "en":
        if not raw_en_cases:
            st.warning(t["no_translation"])
            active_cases = raw_zh_cases
        elif len(raw_en_cases) != len(raw_zh_cases):
            st.warning(t["count_mismatch"].format(en=len(raw_en_cases), zh=len(raw_zh_cases)))
            active_cases = raw_zh_cases
        else:
            active_cases = raw_en_cases
            residual = count_cjk_cases(raw_en_cases)
            if residual:
                st.caption("⚠️ " + t["residual"].format(n=residual))
    else:
        active_cases = raw_zh_cases

    case_labels = [f"{t['case_prefix']}{c.get('case_id', i)}" for i, c in enumerate(active_cases)]

    def on_case_change():
        st.session_state.selected_case_idx = case_labels.index(st.session_state["case_selector"])

    st.selectbox(
        t["select_case"],
        case_labels,
        index=st.session_state.selected_case_idx,
        key="case_selector",
        on_change=on_case_change,
    )

    case_idx = st.session_state.selected_case_idx
    case = active_cases[case_idx]
    zh_case = raw_zh_cases[case_idx]

    playback_delay = st.slider(t["speed"], min_value=0.5, max_value=4.0, value=1.5, step=0.25)

    st.divider()
    board_container = st.container()



if lang_key == "zh":
    milestones = parse_zh_milestones(zh_case.get("milestones", []))
else:
    en_m = case.get("milestones", [])
    if isinstance(en_m, list) and len(en_m) > 0:
        milestones = [str(m).strip(" '\"[]") for m in en_m]
    else:
        milestones = parse_zh_milestones(zh_case.get("milestones", []))

milestone_hits = {int(idx): int(turn) for idx, turn in case.get("milestone_history", [])}

law_hit = case.get("law_found_turn")
law_hit = int(law_hit) if law_hit is not None and int(law_hit) >= 0 else None

messages = case.get("messages", [])


def content_to_str(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def turns_completed(current_message):
    return sum(
        1 for i, m in enumerate(messages) if m.get("role") == "assistant" and i < current_message
    )

"""
Function that draws the board

Parameters:
container - the streamlit container item
current_turn - the current turn of the recorded debate
"""
def render_board(container, current_turn):
    with container:
        container.empty()
        st.subheader(t["milestones"])
        milestone_checked = lambda k: k in milestone_hits and milestone_hits[k] <= current_turn

        for i, m in enumerate(milestones):
            st.markdown(f"{'✅' if milestone_checked(i) else '⬜'} {m}")

        st.divider()
        law_checked = law_hit is not None and law_hit <= current_turn
        st.markdown(f"{'⚖️✅' if law_checked else '⚖️⬜'} {t['law_found_title']}")

        completed = sum(1 for i in range(len(milestones)) if milestone_checked(i))
        total_items = max(len(milestones) + 1, 1)
        st.progress((completed + int(law_checked)) / total_items)

"""
Function that draws new messages based on the turn

Parameters:
container - the streamlit container item
current_msg_idx - the current message turn iterated

"""
def render_chat(container, current_msg_idx):
    with container:
        for m in messages[:current_msg_idx]:
            role = m.get("role", "user")
            content = content_to_str(m.get("content", ""))
            if role == "system":
                continue
            if content.startswith(DB_PREFIX):
                with st.expander(t["db_result"], expanded=False):
                    clean = content.replace(DB_PREFIX + ":", "").replace(DB_PREFIX, "").strip()
                    st.markdown(clean)
                continue
            with st.chat_message(role, avatar=AVATARS.get(role, "❔")):
                st.markdown(content)



st.header(f"{t['case_prefix']}{case.get('case_id', '')}")
with st.expander(t["case_facts"], expanded=False):
    st.write(case.get("fact", ""))

play_col, slider_col = st.columns([1, 4])
with play_col:
    play_button = st.button(t["play"], use_container_width=True)

chat_placeholder = st.empty()

if play_button:
    for k in range(1, len(messages) + 1):
        with chat_placeholder.container():
            render_chat(st.container(), k)
        render_board(board_container, turns_completed(k))
        time.sleep(playback_delay)
else:
    with slider_col:
        current = st.slider(t["reveal"], 0, len(messages), len(messages), label_visibility="collapsed")
    with chat_placeholder.container():
        render_chat(st.container(), current)
    render_board(board_container, turns_completed(current))



# Set up footer that shows metrics
st.divider()
mc = case.get("metrics", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric(t["m_milestones"], f"{mc.get('milestone_frac', 0):.0%}")
c2.metric(t["m_law_found"], "✓" if mc.get("law_found", 0) > 0 else "✗")
c3.metric(t["m_searches"], int(mc.get("searches", 0)))
c4.metric(t["m_reward"], f"{mc.get('reward', 0.0):.2f}" if "reward" in mc else "—")