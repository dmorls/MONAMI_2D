"""Page 3: Algorithm selection and configuration."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.invalidation import (
    commit_algorithm_fingerprint,
    make_algorithm_fingerprint,
    set_current_algorithm_fingerprint,
)
from app.state import algorithm_ready, init_session_state, sampling_ready
from monami.algorithms.registry import get_algorithm, list_algorithms, resolve_algorithm_id

init_session_state()

st.title("3. Algorithm")

if not sampling_ready():
    st.warning("Complete sampling on the **Sampling** page first.")
    st.stop()

if algorithm_ready():
    algo = get_algorithm(st.session_state.selected_algorithm_id)
    st.info(f"**Algorithm ready:** {algo.name}. Change settings and click **Apply algorithm** to refresh.")
elif st.session_state.get("_committed_algorithm_fingerprint") is not None:
    st.warning(
        "Algorithm settings changed since the last apply. "
        "Click **Apply algorithm** before training."
    )

algorithms = list_algorithms()
algo_by_id = {algo.id: algo for algo in algorithms}
algo_ids = [algo.id for algo in algorithms]
current_id = resolve_algorithm_id(
    st.session_state.get("selected_algorithm_id", algo_ids[0])
)
if current_id not in algo_by_id:
    current_id = algo_ids[0]

selected_id = st.selectbox(
    "Prediction algorithm",
    algo_ids,
    index=algo_ids.index(current_id),
    format_func=lambda aid: algo_by_id[aid].name,
)

selected_algo = algo_by_id[selected_id]
st.markdown(selected_algo.long_description or selected_algo.description)

samples = st.session_state.samples_df
categorized = st.session_state.categorized_2d

algo_config = selected_algo.render_config_ui(
    st,
    samples,
    categorized,
    random_seed=int(st.session_state.random_seed),
    default_config=st.session_state.get("algorithm_config"),
)

st.session_state.selected_algorithm_id = selected_id
st.session_state.algorithm_config = algo_config

fingerprint = make_algorithm_fingerprint(selected_id, algo_config)
set_current_algorithm_fingerprint(fingerprint)

if st.button("Apply algorithm", type="primary"):
    commit_algorithm_fingerprint(fingerprint)
    st.success(f"Applied **{selected_algo.name}**.")
    st.rerun()

if algorithm_ready():
    st.caption(selected_algo.feature_summary(algo_config))
