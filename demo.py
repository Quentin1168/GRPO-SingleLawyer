import streamlit as st
import json
import time
st.set_page_config

st.set_page_config(page_title="GRPO-Lawyer Demo", layout="wide")

DEMO_FILE = "demo.jsonl"
AVATARS = {"assistant": "🧑‍⚖️", "user": "🧑‍💼", "system": "⚙️"}

@st.cache_data
def load_cases(path=DEMO_FILE):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def get_milestones(milestone_raw):
    # get the string milestone list and extract milestones

    return [m.strip() for m in milestone_raw.split()]

def render_board(container, current_turn):

    with container:
        st.subheader("Milestones:")
        milestone_checked = lambda k: k in milestone_hits and milestone_hits[k] < current_turn
        for i, m in enumerate(milestones):
            st.markdown(f"{'✅' if milestone_checked(i) else '⬜'} {m}")

        st.divider()
        law_checked = law_hit is not None and law_hit < current_turn
        st.markdown(f"{'⚖️✅' if law_checked else '⚖️⬜'}")
        completed_milestones = sum(1 for i in range(len(milestones)) if milestone_checked(i))

        total_items = max(len(milestones)+1, 1)

        st.progress((completed_milestones + int(law_checked)) / total_items)

def render_chat(area, current_message):
    with area:
        for m in messages[:current_message]:
            role, content = m["role"], m["content"]
            if role == "system":
                continue
            if content.startswith("[Database Result]"):
                with st.expander("🔍 数据库检索结果 / Law search result", expanded=False):
                    st.markdown(content.removeprefix("[Database Result]:").strip())
                continue

            with st.chat_message(role, avatar=AVATARS.get(role, "❔")):
                st.markdown(content)

def turns_completed(current_message):
    return sum(1 for p in [i for i, j in enumerate(messages) if j["role"] == "assistant"] if p < current_message)

cases = load_cases()

with st.sidebar:
    st.title("GRPO-Lawyer")
    case = st.selectbox(
        "Select case / 选择案例",
        cases)
    st.divider()
    board = st.container()

milestones = get_milestones(case["milestones"])
milestone_hits = {int(idx): int(turn) for idx, turn in case["milestone_history"]}

law_hit = case.get("law_found_turn")
if int(law_hit) < 0:
    law_hit = None
else:
    law_hit = int(law_hit)


#load the model dialogue

messages = case["messages"]
total_turns = len([i for i, j in enumerate(messages) if j["role"] == "assistant"])

st.header(f"案例 #{case['case_id']}")
with st.expander("案情 / Case facts", expanded=False):
    st.write(case["fact"])

play, slider = st.columns([1,4])
with play:
    play_button = st.button("▶ 播放 / Play", use_container_width=True)
with slider:
    current = st.slider("Reveal Messages", 0, len(messages), len(messages), label_visibility="collapsed")

chat_area = st.container()

if play_button:
    realtime_chat = st.empty()
    for k in range(1, len(messages) + 1):
        render_chat(st.container(), k)
    board.empty()
    render_board(board, turns_completed(k))

    time.sleep(0.5)

else:
    render_chat(chat_area, current)

st.divider()
mc = case["metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Milestones", f"{mc.get('milestone_frac', 0):.0%}")
c2.metric("Law found", "✓" if mc.get("law_found", 0) > 0 else "✗")
c3.metric("Searches", int(mc.get("searches", 0)))
c4.metric("Reward", f"{mc.get('reward', 0.0):.2f}" if "reward" in mc else "—")