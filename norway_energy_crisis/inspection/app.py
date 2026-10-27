"""Streamlit entry point: python -m streamlit run this_file.py."""

import argparse
import json
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import sqlite3
from zoneinfo import ZoneInfo

from norway_energy_crisis.inspection.details import (
    household_history,
    household_index,
    household_timeline,
)

from norway_energy_crisis.inspection.store import (
    DATABASE,
    diagnostics,
    discover,
    episodes,
    load_episode,
    validation_history,
)


@st.cache_data(max_entries=12, show_spinner=False)
def cached_episode(path, episode_id, revision):
    return load_episode(path, episode_id)


def revision(path):
    paths = [Path(path), Path(str(path) + "-wal")]
    return tuple(
        (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else (0, 0)
        for p in paths
    )


def plot(figure, key):
    figure.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=35, b=15))
    st.plotly_chart(figure, width="stretch", key=key)


def has_allocation(episode):
    """Regional stress limits do not imply that household demand was curtailed."""
    return bool(
        episode["metadata"].get(
            "allocation_enabled", len(episode["context"]["capacity_kw"]) > 0
        )
    )


def overview(folder, frame, kind, history):
    complete = (
        frame[(frame.status == "complete") & (frame.discarded == 0)]
        if not frame.empty
        else frame
    )
    if not complete.empty:
        label = "Training" if kind == "training" else "Checkpoint replay"
        a, b, c, d = st.columns(4)
        a.metric(f"{label} episodes", len(complete))
        b.metric("Highest episode reward", f"{complete.reward.max():.5f}")
        c.metric("Mean episode reward", f"{complete.reward.mean():.5f}")
        d.metric("Latest complete reward", f"{complete.reward.iloc[-1]:.5f}")
        window = st.slider(
            "Reward rolling window",
            1,
            max(2, min(500, len(complete))),
            min(20, len(complete)),
        )
        fig = go.Figure()
        x = complete.last_decision if kind == "training" else complete.id
        for name, values in {
            "Episode reward": complete.reward,
            "Rolling mean": complete.reward.rolling(window, min_periods=1).mean(),
            "Highest so far": complete.reward.cummax(),
        }.items():
            fig.add_scatter(
                x=x,
                y=values,
                name=name,
                mode="lines+markers" if len(complete) < 5 else "lines",
            )
        fig.update_layout(
            xaxis_title=(
                "Policy decisions" if kind == "training" else "Replay episode ID"
            ),
            yaxis_title="Episode reward (higher is better)",
        )
        plot(fig, "overall_rewards")
        if kind != "training":
            st.info(
                "Deterministic checkpoint replays; original training episodes "
                "were not recorded."
            )
    if not history.empty and "validation_reward" in history:
        fig = go.Figure(
            go.Scatter(
                x=history.seconds / 60,
                y=history.validation_reward,
                mode="lines+markers",
                name="Mean validation reward",
            )
        )
        fig.update_layout(
            xaxis_title="Consumed training budget (minutes)",
            yaxis_title="Mean validation reward = −100 × cost/loss",
        )
        plot(fig, "validation_curve")
        st.caption(
            "Validation means are separate from individual training rewards. "
            "Best validation and highest training reward can select different policies."
        )
    csv = folder / "progress.csv"
    if csv.exists():
        with st.expander("Optimizer, exploration and learning-rate logs"):
            try:
                progress = pd.read_csv(csv)
                columns = [c for c in progress if c.startswith(("train/", "rollout/"))]
                chosen = st.multiselect("Logged signals", columns, default=columns[:2])
                if chosen:
                    st.line_chart(progress.set_index("budget/seconds")[chosen])
                st.dataframe(progress, width="stretch")
            except (ValueError, pd.errors.ParserError) as error:
                st.warning(f"The log is currently being written: {error}")


def household_detail(episode, scope):
    """Independent household/hour cursor; refresh does not move an old cursor."""
    context, metadata = episode["context"], episode["metadata"]
    ids = context["household_ids"].astype(str).tolist()
    household_key = f"detail_household:{scope}"
    hour_key = f"detail_hour_widget:{scope}"
    cursor_key = f"{scope}:{episode['id']}"
    # Streamlit removes widget keys when their episode is no longer rendered.
    # Keep remembered cursors separately from widget lifecycle state.
    cursors = st.session_state.setdefault("household_hour_positions", {})
    if st.session_state.get(household_key) not in ids:
        st.session_state[household_key] = ids[0]
    count = len(episode["steps"])
    if not 1 <= cursors.get(cursor_key, 0) <= count:
        cursors[cursor_key] = 1
    position = ids.index(st.session_state[household_key])
    hour = int(cursors[cursor_key])
    st.session_state[hour_key] = hour

    def select_household(offset):
        st.session_state[household_key] = ids[position + offset]

    def select_hour(value):
        cursors[cursor_key] = value
        st.session_state[hour_key] = value

    def remember_hour():
        cursors[cursor_key] = int(st.session_state[hour_key])

    with st.expander("Household step details", expanded=True):
        choices = st.columns([3, 1.3, 1.3])
        household = choices[0].selectbox(
            "Household",
            ids,
            key=household_key,
            format_func=lambda value: (
                f"{value} · {metadata['regions'][context['regions'][ids.index(value)]]}"
            ),
        )
        choices[1].button(
            "Previous household",
            disabled=position == 0,
            on_click=select_household,
            args=(-1,),
        )
        choices[2].button(
            "Next household",
            disabled=position == len(ids) - 1,
            on_click=select_household,
            args=(1,),
        )
        navigation = st.columns(4)
        for column, label, value in zip(
            navigation,
            ("First hour", "Previous hour", "Next hour", "Last recorded hour"),
            (1, max(1, hour - 1), min(count, hour + 1), count),
        ):
            column.button(
                label, disabled=value == hour, on_click=select_hour, args=(value,)
            )
        hour = int(
            st.number_input(
                "Inspect hour",
                min_value=1,
                max_value=count,
                step=1,
                key=hour_key,
                on_change=remember_hour,
            )
        )
        timeline = household_timeline(episode, household)
        row = timeline.iloc[hour - 1]
        h = household_index(episode, household)
        timestamp = pd.Timestamp(metadata["origin"]) + pd.Timedelta(
            hours=(hour - 1) * metadata["interval_hours"]
        )
        st.caption(
            f"Episode #{episode['id']} · household {household} · hour {hour}/{count} "
            f"recorded · {timestamp:%d %b %Y %H:%M} (dataset local time). "
            "Stepping changes this view only; training continues independently."
        )
        a, b, c = st.columns(3)
        a.metric(
            "Household power (kW)",
            f"{row.requested_kw:.4f}",
            delta=f"{row.power_change_kw:+.4f} kW from preceding hour",
            delta_color="off",
        )
        b.metric("Raw prediction (kW)", f"{row.raw_kw:.4f}")
        c.metric("Reference power (kW)", f"{row.reference_kw:.4f}")
        a, b, c = st.columns(3)
        a.metric("Step price (NOK/kWh)", f"{row.price_nok_kwh:.2f}")
        b.metric("Household energy so far (kWh)", f"{row.cumulative_requested_kwh:.4f}")
        c.metric(
            "Household charge so far (NOK)",
            f"{row.cumulative_household_charge_nok:.4f}",
        )
        if has_allocation(episode):
            a, b, c = st.columns(3)
            a.metric("Delivered power (kW)", f"{row.delivered_kw:.4f}")
            b.metric("Essential requirement (kW)", f"{row.essential_kw:.4f}")
            c.metric(
                "Unserved energy so far (kWh)", f"{row.cumulative_unserved_kwh:.4f}"
            )
        fig = go.Figure()
        dt = metadata["interval_hours"]
        initial = context["initial_history"][h]
        fig.add_scatter(
            x=np.arange(1 - len(initial), 1),
            y=initial / dt,
            name="Starting history",
            line=dict(color="gray"),
        )
        for name in ("requested", "raw", "reference") + (
            ("delivered",) if has_allocation(episode) else ()
        ):
            fig.add_scatter(
                x=timeline.hour, y=timeline[f"{name}_kw"], name=name.title()
            )
        fig.add_vline(
            x=hour, line_dash="dash", annotation_text=f"Inspecting hour {hour}"
        )
        fig.add_scatter(
            x=[hour],
            y=[row.requested_kw],
            mode="markers",
            marker=dict(size=11),
            name="Selected step",
        )
        fig.update_layout(
            height=330,
            xaxis_title="Relative hour",
            yaxis_title="Household average power (kW)",
        )
        plot(fig, "household_detail_curve")
        st.write(
            f"**Shared action {int(episode['steps'].action.iloc[hour-1])} · "
            f"cohort step reward {row.cohort_reward:.6f} · "
            f"cohort reward so far {row.cumulative_cohort_reward:.6f}**"
        )
        st.caption(
            "The reward applies to all households together; "
            "no individual-household reward is defined."
        )
        if "load_ratio_0" in episode["steps"]:
            region = int(context["regions"][h])
            step = episode["steps"].iloc[hour - 1]
            st.caption(
                f"Regional load/capacity: {step[f'load_ratio_{region}']:.3f} · "
                f"pre-action grid weight: {step[f'grid_weight_{region}']:.3f} · "
                f"unweighted stress cost: {step[f'stress_cost_{region}']:.5f}"
            )
        components = episode["steps"].filter(regex="^reward_").iloc[hour - 1]
        st.dataframe(
            components.rename("Booked cohort reward").to_frame(), width="stretch"
        )
        with st.expander("Exact household sequence and incoming demand history"):
            st.dataframe(timeline, width="stretch", hide_index=True)
            st.download_button(
                "Download selected household CSV",
                timeline.to_csv(index=False),
                f"episode_{episode['id']}_household_{household}.csv",
                "text/csv",
            )
            st.caption(
                "Demand history before this action: only starting values "
                "and earlier simulated requests."
            )
            st.dataframe(
                household_history(episode, household, hour),
                width="stretch",
                hide_index=True,
            )


def inspector(episode, scope):
    frame, arrays, context = episode["steps"], episode["arrays"], episode["context"]
    metadata = episode["metadata"]
    st.write(f"**{metadata['origin']} · {metadata['scenario']} · {episode['status']}**")
    household_detail(episode, scope)
    st.caption(
        metadata["reward_timing"]
        + ". Components show booked contributions, not causal credit for an action."
    )
    if "service" in " ".join(frame.columns) or "reward_energy" in frame:
        st.info(
            "Terminal penalties evaluate the whole trajectory. The final action "
            "does not independently cause the whole daily penalty."
        )
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Price decisions",
            "Reward booked at each hour",
            "Cumulative reward",
        ),
        vertical_spacing=0.1,
    )
    fig.add_scatter(
        x=frame.hour,
        y=frame.price,
        name="Selected price",
        line_shape="hv",
        row=1,
        col=1,
    )
    for area, name in enumerate(metadata["regions"]):
        fig.add_scatter(
            x=np.arange(1, context["market_prices"].shape[1] + 1),
            y=context["market_prices"][area],
            name=f"Recorded {name} price",
            line=dict(dash="dot"),
            row=1,
            col=1,
        )
    for column in frame.filter(regex="^reward_"):
        fig.add_trace(
            go.Bar(x=frame.hour, y=frame[column], name=column.removeprefix("reward_")),
            row=2,
            col=1,
        )
    fig.add_scatter(
        x=frame.hour, y=frame.reward.cumsum(), name="Cumulative reward", row=3, col=1
    )
    fig.update_layout(height=650, barmode="relative")
    fig.update_yaxes(title_text="NOK/kWh", row=1, col=1)
    fig.update_xaxes(title_text="Episode hour", row=3, col=1)
    plot(fig, "episode_rewards")
    if "load_ratio_0" in frame:
        stress_plot = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Regional load / constructed capacity",
                "Pre-action grid weights",
            ),
        )
        for area, name in enumerate(metadata["regions"]):
            stress_plot.add_scatter(
                x=frame.hour,
                y=frame[f"load_ratio_{area}"],
                name=f"{name} load ratio",
                row=1,
                col=1,
            )
            stress_plot.add_scatter(
                x=frame.hour,
                y=frame[f"grid_weight_{area}"],
                name=f"{name} weight",
                row=2,
                col=1,
            )
        stress_plot.add_hline(y=1, line_dash="dash", row=1, col=1)
        stress_plot.add_hline(y=0.9, line_dash="dot", row=1, col=1)
        stress_plot.update_layout(height=450)
        plot(stress_plot, "regional_stress_weights")
        with st.expander("Recorded weights and unweighted stress costs"):
            columns = ["hour"] + [
                c
                for c in frame
                if c.startswith(
                    (
                        "load_ratio_",
                        "previous_ratio_",
                        "grid_weight_",
                        "stress_cost_",
                        "reward_grid_",
                    )
                )
            ]
            st.dataframe(frame[columns], width="stretch", hide_index=True)
    dt = metadata["interval_hours"]
    regions = metadata["regions"]
    region = st.selectbox("Demand region", ["All households", *regions])
    mask = (
        np.ones(len(context["household_ids"]), bool)
        if region == "All households"
        else context["regions"] == regions.index(region)
    )
    fig = go.Figure()
    for key, label in (
        ("requested", "Requested / predicted demand"),
        ("delivered", "Delivered by simulator"),
    ):
        if key == "delivered" and not has_allocation(episode):
            continue
        fig.add_scatter(x=frame.hour, y=arrays[key][:, mask].sum(1) / dt, name=label)
    fig.add_scatter(
        x=np.arange(1, len(context["reference"]) + 1),
        y=context["reference"][:, mask].sum(1) / dt,
        name="Modeled recorded-price reference",
        line=dict(dash="dash"),
    )
    if len(context["capacity_kw"]) and region != "All households":
        fig.add_hline(
            y=context["capacity_kw"][regions.index(region)],
            line_dash="dot",
            annotation_text="Constructed regional capacity",
        )
    fig.update_layout(
        xaxis_title="Episode hour", yaxis_title="Hourly average power (kW)"
    )
    plot(fig, "aggregate_demand")
    if region == "All households" and len(context["capacity_kw"]):
        st.caption(
            "NO1 and NO5 are separate capacity pools. Select a region to inspect "
            "its own limit; the combined total is descriptive."
        )
    ids = context["household_ids"]
    available = ids[mask].tolist()
    biggest = ids[
        np.flatnonzero(mask)[np.argsort(arrays["requested"][:, mask].sum(0))[-3:]]
    ].tolist()
    selected = st.multiselect(
        "Households to inspect", available, default=biggest, max_selections=8
    )
    quantity = st.selectbox(
        "Household quantity",
        ["requested", "raw"]
        + (["delivered", "essential"] if has_allocation(episode) else []),
        format_func=lambda key: {
            "requested": "Requested demand (after any external floor)",
            "raw": "Raw consumer-model prediction",
            "delivered": "Delivered by the simulator",
            "essential": "Assumed essential requirement",
        }[key],
    )
    fig = go.Figure()
    for household in selected:
        h = int(np.flatnonzero(ids == household)[0])
        if quantity in ("requested", "raw"):
            values = np.r_[context["initial_history"][h], arrays[quantity][:, h]] / dt
            hours = np.arange(-23, len(frame) + 1)
        else:
            # Actual pre-episode allocations/essential loads were not observed.
            values, hours = arrays[quantity][:, h] / dt, frame.hour
        fig.add_scatter(x=hours, y=values, name=f"Household {household}")
    fig.add_vline(x=0.5, line_dash="dash")
    fig.update_layout(
        xaxis_title="Hour (≤0: starting history; ≥1: simulated)",
        yaxis_title="Hourly average power (kW)",
    )
    plot(fig, "household_trends")
    if st.checkbox("Show all-household demand heatmap"):
        fig = go.Figure(
            go.Heatmap(
                x=frame.hour,
                y=ids[mask],
                z=arrays[quantity][:, mask].T / dt,
                colorbar=dict(title="kW"),
            )
        )
        fig.update_layout(
            height=550, xaxis_title="Episode hour", yaxis_title="Household ID"
        )
        plot(fig, "household_heatmap")
    with st.expander("Exact action/reward table and downloads"):
        st.dataframe(frame.drop(columns="episode"), width="stretch")
        st.download_button(
            "Download hourly action/reward CSV",
            frame.to_csv(index=False),
            "episode_actions.csv",
            "text/csv",
        )
        if st.checkbox("Prepare every-household hourly CSV"):
            hourly = pd.DataFrame(
                {
                    "hour": np.repeat(frame.hour.to_numpy(), len(ids)),
                    "household_id": np.tile(ids, len(frame)),
                    "price_nok_kwh": np.repeat(frame.price.to_numpy(), len(ids)),
                    **{f"{name}_kwh": value.ravel() for name, value in arrays.items()},
                }
            )
            st.download_button(
                "Download all household trajectories",
                hourly.to_csv(index=False),
                "episode_households.csv",
                "text/csv",
            )


def current_state(episode):
    frame, arrays, context = episode["steps"], episode["arrays"], episode["context"]
    st.caption(
        f"Latest recorded episode #{episode['id']}, "
        f"{episode['metadata']['origin']}, {episode['status']}. "
        "Simulator values, not live household meters."
    )
    hour = (
        st.slider("Recorded hour", 1, max(2, len(frame)), len(frame))
        if len(frame) > 1
        else 1
    )
    i = min(hour, len(frame)) - 1
    dt = episode["metadata"]["interval_hours"]
    previous = (
        context["initial_history"][:, -1] if i == 0 else arrays["requested"][i - 1]
    )
    table = pd.DataFrame(
        {
            "Household": context["household_ids"],
            "Region": [episode["metadata"]["regions"][a] for a in context["regions"]],
            "Requested power kW": arrays["requested"][i] / dt,
            "Delivered power kW": arrays["delivered"][i] / dt,
            "Reference power kW": context["reference"][i] / dt,
            "Change from preceding hour kW": (arrays["requested"][i] - previous) / dt,
            "Cumulative requested kWh": arrays["requested"][: i + 1].sum(0),
            "Cumulative delivered kWh": arrays["delivered"][: i + 1].sum(0),
            "Modeled energy charge NOK": (
                arrays["delivered"][: i + 1] * frame.price.to_numpy()[: i + 1, None]
            ).sum(0),
            "Forced unserved energy kWh": (
                arrays["requested"][: i + 1] - arrays["delivered"][: i + 1]
            ).sum(0),
        }
    )
    if not has_allocation(episode):
        table = table.drop(
            columns=[
                "Delivered power kW",
                "Cumulative delivered kWh",
                "Forced unserved energy kWh",
            ]
        )
    a, b, c = st.columns(3)
    a.metric("Total requested power", f"{table['Requested power kW'].sum():,.2f} kW")
    b.metric("Price (NOK/kWh)", f"{frame.price.iloc[i]:.2f}")
    c.metric("Reward so far", f"{frame.reward.iloc[:i+1].sum():.5f}")
    st.dataframe(
        table.sort_values("Requested power kW", ascending=False),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download household state CSV",
        table.to_csv(index=False),
        "household_state.csv",
        "text/csv",
    )


def render(folder):
    path = folder / DATABASE
    history_folder = folder
    if not path.exists():
        settings = folder / "settings.json"
        if settings.exists() and str(
            json.loads(settings.read_text()).get("version", "")
        ).startswith("regional_stress_"):
            status_file = folder / "status.json"
            status = (
                json.loads(status_file.read_text())
                if status_file.exists()
                else {"stage": "initializing"}
            )
            st.info(f"Pricing experiment: {status.get('stage','unknown')}")
            st.json(status)
            marker = folder / "active_run.json"
            run = (
                json.loads(marker.read_text()).get("run")
                if marker.exists()
                else status.get("run")
            )
            if run:
                active = (folder / run).resolve()
                if active.parent == folder.resolve() and (active / DATABASE).exists():
                    render(active)
            return
        st.warning(
            "This older run has validation summaries but no individual training "
            "traces. Original household/action histories cannot be reconstructed."
        )
        overview(folder, pd.DataFrame(), "training", validation_history(folder))
        st.code(
            f'python -m norway_energy_crisis.inspection.replay --run "{folder}" '
            "--days 3 --device cpu"
        )
        return
    frame = episodes(path)
    if frame.empty:
        st.info("Recording is ready. Actions will appear when training starts.")
        return
    kinds = frame.kind.unique().tolist()
    kind = st.radio(
        "Episode source",
        kinds,
        horizontal=True,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    include_discarded = st.checkbox(
        "Include interactions discarded after checkpoint recovery"
    )
    frame = frame[frame.kind.eq(kind)]
    if not include_discarded:
        frame = frame[frame.discarded.eq(0)]
    if frame.empty:
        st.info("No episodes in this filter.")
        return
    latest = cached_episode(str(path), int(frame.id.iloc[-1]), revision(path))
    st.caption(
        f"Latest recording: episode #{latest['id']}, "
        f"hour {latest['steps'].hour.iloc[-1]}"
        f" · {latest['status']}. Updates include partially recorded episodes."
    )
    history_folder = Path(latest["manifest"].get("source_run", folder))
    tabs = st.tabs(
        ["Reward trends", "Episode inspector", "Latest household state", "Diagnostics"]
    )
    with tabs[0]:
        overview(history_folder, frame, kind, validation_history(history_folder))
    with tabs[1]:
        choices = frame.id.tolist()
        key = f"episode:{folder}:{kind}"
        follow_key = f"follow:{folder}:{kind}"
        labels = frame.set_index("id")

        def choose_best():
            complete = frame[frame.status.eq("complete")]
            if not complete.empty:
                st.session_state[key] = int(
                    complete.loc[complete.reward.idxmax(), "id"]
                )
                st.session_state[follow_key] = False

        st.button("Inspect highest-reward complete episode", on_click=choose_best)
        follow = st.checkbox("Follow newest episode", key=follow_key)
        if follow or st.session_state.get(key) not in choices:
            st.session_state[key] = choices[-1]
        chosen = st.selectbox(
            "Episode",
            choices,
            key=key,
            disabled=follow,
            format_func=lambda i: (
                f"#{i} · {labels.loc[i, 'origin']} · "
                f"reward {labels.loc[i, 'reward']:.5f} · {labels.loc[i, 'status']}"
            ),
        )
        selected = cached_episode(str(path), int(chosen), revision(path))
        inspector(selected, str(folder))
    with tabs[2]:
        current_state(latest)
    with tabs[3]:
        st.write(f"**Selected episode #{chosen}**")
        for level, message in diagnostics(selected):
            getattr(st, level)(message)
        steps = selected["steps"]
        grid = selected["context"]["grid"]
        occupancy = steps.action.value_counts().reindex(range(len(grid)), fill_value=0)
        fig = go.Figure(go.Bar(x=grid, y=occupancy.to_numpy()))
        fig.update_layout(
            xaxis_title="Price (NOK/kWh)", yaxis_title="Actions in selected episode"
        )
        plot(fig, "action_occupancy")
        totals = steps.filter(regex="^reward_").sum().rename("Booked reward")
        st.dataframe(totals.to_frame(), width="stretch")
        st.caption(
            "These patterns do not automatically prove a faulty policy. "
            "Reference curves are modeled, not measured counterfactual demand."
        )
        with st.expander("Episode provenance"):
            st.json({"episode": selected["metadata"], "run": selected["manifest"]})


def main():
    st.set_page_config(page_title="Electricity training inspector", layout="wide")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path(
            os.environ.get(
                "NORWAY_DASHBOARD_RUNS", Path(__file__).resolve().parents[1] / "runs"
            )
        ),
    )
    args, _ = parser.parse_known_args()
    st.title("Electricity training inspector")
    st.caption(
        "Trace each action, household prediction and reward. "
        "Training and checkpoint replays remain separate."
    )
    root = (
        Path(st.sidebar.text_input("Runs folder", str(args.runs)))
        .expanduser()
        .resolve()
    )
    controls = st.columns([1.1, 1.4, 1])
    auto = controls[0].checkbox("Automatic refresh", value=True)
    interval = controls[1].selectbox(
        "Refresh interval",
        [5, 15, 30, 60, 120],
        index=2,
        format_func=lambda seconds: f"Every {seconds} seconds",
        disabled=not auto,
    )
    if controls[2].button("Refresh now", width="stretch"):
        cached_episode.clear()
    st.caption(
        "Use alongside training. Refresh reads saved steps without pausing the fit."
    )
    st.sidebar.caption(
        "The viewer performs no training or inference. "
        "Every recorded episode is retained on disk."
    )

    @st.fragment(run_every=interval if auto else None)
    def live():
        try:
            st.caption(
                "Last refreshed: "
                + datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
                    "%d %b %Y %H:%M:%S IST"
                )
            )
            folders = discover(root)
            if not folders:
                st.info("No policy runs yet. Refresh will discover new training runs.")
                return
            run_key = f"selected_run:{root}"
            if st.session_state.get(run_key) not in folders:
                st.session_state[run_key] = next(
                    (p for p in folders if (p / DATABASE).exists()), folders[0]
                )
            folder = st.selectbox(
                "Run",
                folders,
                key=run_key,
                format_func=lambda p: p.relative_to(root).as_posix(),
            )
            render(folder)
        except (OSError, ValueError, KeyError, sqlite3.Error) as error:
            st.error(f"Unable to read this run: {error}")

    live()


if __name__ == "__main__":
    main()
