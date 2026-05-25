#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   NEUROCHIP SMARTSILICON FAULT ANALYSIS                      ║
║           GNN-powered fault detection & localization for Verilog             ║
║                      Author: Nikhil Bhaktha                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
─────
  Interactive mode  (prompts for paths):
      python NeuroChip_Detector_CLI.py

  Command-line mode:
      python NeuroChip_Detector_CLI.py --verilog path/to/circuit.v      \
                                       --top-k   10                     \\
                                       --save-report report.json

  Batch mode (multiple files):
      python NeuroChip_Detector_CLI.py --verilog a.v b.v c.v            \
                                       --save-report batch_report.json
"""
# =============================================================================

# Imports

import argparse
import json
import os
import re
import sys
import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless-safe; swap to TkAgg for live windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

warnings.filterwarnings("ignore")

# =============================================================================

# Constants

NUM_NODE_FEATURES = 15          # 11 type one-hots + is_port + fanin/fanout/degree norm
NUM_GRAPH_FEATURES = 6          # signal_map_len, num_gates, num_inputs, num_outputs, avg_degree, max_depth
HIDDEN_DIM = 64
NUM_CLASSES = 2

ALL_GATE_TYPES = [
    "input", "output", "assign", "and", "or",
    "not",   "xor",    "nor",    "xnor","buf",  "inv",
]
GATE_PRIMITIVES = {"and","or","not","buf","xor","nor","xnor","nand","inv"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================

# GNN Model

# Defines a two-stage GNN for graph-level fault classification.
#       Stage 1 — Three GCNConv layers with residual connections and batch norm.
#       Stage 2 — Dual global pooling fused with circuit-level context features, fed into a 3-layer MLP classifier.

class FaultGNN(nn.Module):
    def __init__(self,
                 in_channels: int  = NUM_NODE_FEATURES,
                 hidden: int       = HIDDEN_DIM,
                 graph_feat_dim: int = NUM_GRAPH_FEATURES,
                 num_classes: int  = NUM_CLASSES,
                 dropout: float    = 0.3):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)

        self.res_proj = (nn.Linear(in_channels, hidden)
                         if in_channels != hidden else nn.Identity())

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)

        self.gf_fc = nn.Sequential(
            nn.Linear(graph_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        clf_in = 2 * hidden + 32
        self.clf = nn.Sequential(
            nn.Linear(clf_in, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        _ = self.res_proj(x)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index))) + x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index))) + x

        g_mean = global_mean_pool(x, batch)
        g_max  = global_max_pool(x, batch)
        g = torch.cat([g_mean, g_max], dim=1)

        gf = data.graph_feats
        if gf.dim() == 3:
            gf = gf.squeeze(1)
        gf_emb = self.gf_fc(gf)

        return self.clf(torch.cat([g, gf_emb], dim=1))

    @torch.no_grad()
    def node_fault_scores(self, data: Data) -> torch.Tensor:
        """
        Per-node suspicion score in [0, 1].

        Each node embedding is compared to the graph centroid in GCN embedding
        space. Nodes furthest from the centroid have the highest anomaly score
        and are the prime candidates for fault localisation.
        """
        self.eval()
        x  = data.x.to(DEVICE)
        ei = data.edge_index.to(DEVICE)
        _  = self.res_proj(x)
        x  = F.relu(self.bn1(self.conv1(x, ei)))
        x  = F.relu(self.bn2(self.conv2(x, ei))) + x
        x  = F.relu(self.bn3(self.conv3(x, ei))) + x
        centroid = x.mean(dim=0, keepdim=True)
        scores   = torch.norm(x - centroid, dim=1)
        scores   = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores.cpu()

# =============================================================================

# Fault-Suggestion Knowledge Base

FAULT_SUGGESTIONS: dict[str, list[str]] = {
    "input": [
        "Verify the input is driven by the testbench or parent module.",
        "Check for missing wire declarations or unconnected ports.",
        "Ensure the signal is not left floating (Z-state) during simulation.",
    ],
    "output": [
        "Ensure every output is driven by exactly one assign statement or gate.",
        "Check for multiple drivers — bus contention causes X-state propagation.",
        "Confirm the output is reachable from all primary inputs via valid logic.",
    ],
    "assign": [
        "Review the RHS expression for incorrect operator precedence or missing parentheses.",
        "Check for X/Z propagation from uninitialized or floating inputs.",
        "Verify bitwidth matches across the assignment (truncation/sign-extension bugs).",
    ],
    "buf": [
        "A buffer with fan-out 0 is a dangling net — connect or remove it.",
        "High fan-out buffers can cause timing closure issues; use buffer trees.",
        "Confirm the buffer is not masking an unintended wire-level inversion.",
    ],
    "inv": [
        "Verify the inversion polarity is intentional at this point in the path.",
        "Check for an unintended double-inversion that simplifies to a wire.",
        "Ensure active-low/active-high conventions are respected at this node.",
    ],
    "and": [
        "Confirm all inputs are fully driven before simulation begins.",
        "Check for stuck-at-0: one input permanently low masks the other inputs.",
        "Verify AND-gate enable signals are not inverted relative to the spec.",
    ],
    "or": [
        "Check for stuck-at-1: one input permanently high dominates the output.",
        "Verify the gate is not collapsed to a wire in the single-input edge case.",
        "Confirm OR is not accidentally an XOR after synthesis optimisation.",
    ],
    "xor": [
        "An XOR with one constant input simplifies to BUF or INV — check intent.",
        "Verify parity logic for off-by-one errors in bit indexing.",
        "Check that XNOR/XOR polarity matches the spec (common confusion source).",
    ],
    "nor": [
        "Check for complementary logic errors introduced after DeMorgan transforms.",
        "Verify NOR is not confused with OR in schematics or net-list annotations.",
    ],
    "xnor": [
        "Verify equivalence-checking logic: XNOR outputs 1 only when inputs match.",
        "Check XNOR/XOR polarity in comparators — swapping these is a common bug.",
    ],
    "nand": [
        "Check for stuck-at-1 on the output if any input is always 0.",
        "Verify NAND is not confused with NOR; double-check DeMorgan equivalences.",
    ],
    "default": [
        "Inspect this node's fan-in/fan-out ratio against the design specification.",
        "Run stuck-at-fault simulation (0 and 1) to isolate the fault site.",
        "Check whether this node sits on a critical timing path (setup/hold).",
    ],
}

# =============================================================================

# Verilog Parser

def parseVerilog(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Lightweight structural Verilog parser supporting:
      - module / endmodule declarations
      - input / output / wire declarations (scalar and bus [n:m])
      - gate primitives: and, or, not, buf, xor, nor, xnor, nand, inv
      - continuous assign statements (combinational logic)

    Returns
    -------
    node_df : DataFrame — one row per signal / gate output
        (columns: node_id, signal_name, type, fanin, fanout, is_port, line_no)
    edge_df : DataFrame — one row per signal connection (src → dst)
        (columns: src, dst)
    meta    : dict — module_name, num_inputs, num_outputs, num_gates, num_wires, source_lines (list of (lineno, text))
    """
    signal_to_id: dict[str, int]  = {}
    line_map:     dict[str, int]  = {}
    node_types:   dict[str, str]  = {}
    fanin_cnt:    dict[str, int]  = {}
    fanout_cnt:   dict[str, int]  = {}
    edges:        list[dict]      = []
    source_lines: list[tuple]     = []

    module_name = "unknown"
    num_inputs = num_outputs = num_gates = num_wires = 0

    def get_id(name: str, lineno: int = -1) -> int:
        """Return (creating if needed) a stable integer id for a signal name."""
        if name not in signal_to_id:
            signal_to_id[name] = len(signal_to_id)
            line_map[name] = lineno
        return signal_to_id[name]

    def add_edge(src_name: str, dst_name: str, lineno: int):
        get_id(src_name, lineno)
        get_id(dst_name, lineno)
        edges.append({"src": signal_to_id[src_name],
                      "dst": signal_to_id[dst_name]})
        fanout_cnt[src_name] = fanout_cnt.get(src_name, 0) + 1
        fanin_cnt[dst_name]  = fanin_cnt.get(dst_name, 0)  + 1

    try:
        raw_text = Path(filepath).read_text(errors="replace")
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {filepath}")
        return pd.DataFrame(), pd.DataFrame(), {}
    except PermissionError:
        print(f"\n[ERROR] Permission denied reading: {filepath}")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Strip block comments (/* … */) before line-by-line processing
    raw_text = re.sub(r"/\*.*?\*/", " ", raw_text, flags=re.DOTALL)
    lines = raw_text.splitlines()

    for lineno, raw_line in enumerate(lines, start=1):
        # Strip inline comments
        line = re.sub(r"//.*", "", raw_line).strip()
        if not line:
            continue
        source_lines.append((lineno, raw_line.rstrip()))

        # ── module declaration ─────────────────────────────────────────────
        m = re.match(r"module\s+(\w+)\s*[#(]?", line)
        if m:
            module_name = m.group(1)
            continue

        # ── port type declarations (input/output/inout) ───────────────
        # Handles: input a, b; | input [7:0] bus; | input signed [3:0] s;
        m = re.match(r"(input|output|inout)\s+"
                     r"(?:signed\s+)?(?:\[\d+:\d+\]\s+)?"
                     r"([\w,\s]+);", line)
        if m:
            ptype  = m.group(1)
            names  = [s.strip() for s in m.group(2).split(",") if s.strip()]
            for s in names:
                get_id(s, lineno)
                node_types[s] = "input" if ptype == "input" else "output"
                fanin_cnt.setdefault(s, 0)
                fanout_cnt.setdefault(s, 0)
                if ptype == "input":
                    num_inputs += 1
                else:
                    num_outputs += 1
            continue

        # ── wire/reg declarations ────────────────────────────────────────
        m = re.match(r"(?:wire|reg)\s+(?:\[\d+:\d+\]\s+)?([\w,\s]+);", line)
        if m:
            names = [s.strip() for s in m.group(1).split(",") if s.strip()]
            for s in names:
                get_id(s, lineno)
                node_types.setdefault(s, "assign")
                fanin_cnt.setdefault(s, 0)
                fanout_cnt.setdefault(s, 0)
                num_wires += 1
            continue

        # ── Gate primitives ────────────────────────────────────────────────
        # Syntax: <gate_type> [#(delay)] <instance_name> (out, in1, in2, ...);
        # Allows optional strength/delay annotations and multi-line ports.
        gate_matched = False
        for gtype in GATE_PRIMITIVES:
            m = re.match(
                rf"({gtype})\s+(?:#\([^)]*\)\s+)?(\w+)\s*\(([^)]*)\)\s*;",
                line, re.IGNORECASE)
            if m:
                num_gates += 1
                ports  = [p.strip() for p in m.group(3).split(",") if p.strip()]
                if not ports:
                    break
                output_sig = ports[0]
                input_sigs = ports[1:]
                get_id(output_sig, lineno)
                node_types[output_sig] = gtype.lower()
                fanin_cnt.setdefault(output_sig, 0)
                fanout_cnt.setdefault(output_sig, 0)
                for inp in input_sigs:
                    # strip bit-select suffixes: signal[2] → signal
                    inp_clean = re.sub(r"\[.*?\]", "", inp).strip()
                    if inp_clean:
                        node_types.setdefault(inp_clean, "assign")
                        fanin_cnt.setdefault(inp_clean, 0)
                        fanout_cnt.setdefault(inp_clean, 0)
                        add_edge(inp_clean, output_sig, lineno)
                gate_matched = True
                break
        if gate_matched:
            continue

        # ── Continuous assign ──────────────────────────────────────────────
        # Syntax: assign lhs = rhs_expression;
        m = re.match(r"assign\s+([\w\[\]:]+)\s*=\s*(.+?)\s*;", line)
        if m:
            lhs_raw = re.sub(r"\[.*?\]", "", m.group(1)).strip()
            rhs     = m.group(2)
            if lhs_raw:
                get_id(lhs_raw, lineno)
                node_types.setdefault(lhs_raw, "assign")
                fanin_cnt.setdefault(lhs_raw, 0)
                fanout_cnt.setdefault(lhs_raw, 0)
                rhs_signals = re.findall(r"\b([a-zA-Z_]\w*)\b", rhs)
                skip = {"and","or","not","xor","buf","nand","nor","xnor",
                        "if","else","begin","end","always","initial",
                        "posedge","negedge","1","0"}
                for sig in rhs_signals:
                    if sig.lower() not in skip:
                        sig_clean = sig.strip()
                        node_types.setdefault(sig_clean, "assign")
                        fanin_cnt.setdefault(sig_clean, 0)
                        fanout_cnt.setdefault(sig_clean, 0)
                        add_edge(sig_clean, lhs_raw, lineno)

    # ── Build node DataFrame ───────────────────────────────────────────────
    node_rows = []
    for sig, sid in signal_to_id.items():
        node_rows.append({
            "node_id":     sid,
            "signal_name": sig,
            "type":        node_types.get(sig, "assign"),
            "fanin":       fanin_cnt.get(sig,  0),
            "fanout":      fanout_cnt.get(sig, 0),
            "is_port":     int(node_types.get(sig, "") in {"input","output"}),
            "line_no":     line_map.get(sig, -1),
        })
    node_df = pd.DataFrame(node_rows) if node_rows else pd.DataFrame()
    edge_df = (pd.DataFrame(edges) if edges
               else pd.DataFrame(columns=["src","dst"]))

    meta = {
        "module_name": module_name,
        "num_inputs":  num_inputs,
        "num_outputs": num_outputs,
        "num_gates":   num_gates,
        "num_wires":   num_wires,
        "num_signals": len(signal_to_id),
        "source_lines": source_lines,
    }
    return node_df, edge_df, meta

# =============================================================================

# Build PyG Data Object

# Converts parsed Verilog DataFrames to a PyG Data object using exactly the same feature encoding as the training pipeline.
def build_PYGdata(node_df: pd.DataFrame,
                   edge_df: pd.DataFrame,
                   meta: dict) -> Data | None:
    """
    Node features (15-dim):
        [0:11]  One-hot over ALL_GATE_TYPES
        [11]    is_port flag
        [12]    fanin  / max_fanin   (normalised)
        [13]    fanout / max_fanout  (normalised)
        [14]    degree / max_degree  (normalised)
    """
    if node_df.empty or edge_df.empty:
        return None

    type_to_vec = {t: [int(t == tt) for tt in ALL_GATE_TYPES]
                   for t in ALL_GATE_TYPES}

    max_fanin  = max(int(node_df["fanin"].max()),  1)
    max_fanout = max(int(node_df["fanout"].max()), 1)
    max_deg    = max(int((node_df["fanin"] + node_df["fanout"]).max()), 1)

    rows = []
    for _, row in node_df.iterrows():
        t    = str(row["type"]).lower()
        tvec = type_to_vec.get(t, [0] * 11)
        fi   = float(row["fanin"])  / max_fanin
        fo   = float(row["fanout"]) / max_fanout
        deg  = (float(row["fanin"]) + float(row["fanout"])) / max_deg
        rows.append(tvec + [int(row["is_port"]), fi, fo, deg])

    x = torch.tensor(rows, dtype=torch.float)

    # Filter edges (both endpoints must be valid node_ids)
    valid_ids = set(node_df["node_id"].tolist())
    edge_df_clean = edge_df[
        edge_df["src"].isin(valid_ids) & edge_df["dst"].isin(valid_ids)
    ]
    if edge_df_clean.empty:
        return None

    edge_index = torch.tensor(
        [edge_df_clean["src"].tolist(), edge_df_clean["dst"].tolist()],
        dtype=torch.long)

    avg_deg = float((node_df["fanin"] + node_df["fanout"]).mean())
    graph_feats = torch.tensor([[
        float(meta.get("num_signals", 0)),
        float(meta.get("num_gates",   0)),
        float(meta.get("num_inputs",  0)),
        float(meta.get("num_outputs", 0)),
        avg_deg,
        0.0,  # max_depth
    ]], dtype=torch.float)

    batch = torch.zeros(len(node_df), dtype=torch.long)
    return Data(x=x, edge_index=edge_index,
                graph_feats=graph_feats, batch=batch)

# ============================================================================

# Model Loader

def load_model(model_path: str | None = None) -> FaultGNN:
    if not model_path:
        model_path = str(Path(__file__).resolve().with_name("NeuroChip.pt"))
    else:
        candidate = Path(model_path).expanduser()
        if not candidate.is_absolute():
            script_candidate = Path(__file__).resolve().parent / candidate
            if script_candidate.exists():
                candidate = script_candidate
        model_path = str(candidate.resolve())

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = FaultGNN(
        in_channels    = NUM_NODE_FEATURES,
        hidden         = HIDDEN_DIM,
        graph_feat_dim = NUM_GRAPH_FEATURES,
        num_classes    = NUM_CLASSES,
    ).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

# =============================================================================

# Visualization Helpers

NODE_COLORS = {
    "input":  "#4CAF50",  # green
    "output": "#F44336",  # red
    "assign": "#2196F3",  # blue
    "and":    "#795548",  # brown
    "or":     "#00BCD4",  # cyan
    "not":    "#607D8B",  # blue-grey
    "inv":    "#9C27B0",  # purple
    "buf":    "#FF9800",  # orange
    "xor":    "#E91E63",  # pink
    "nor":    "#FFEB3B",  # yellow
    "xnor":   "#8BC34A",  # light-green
    "nand":   "#FF5722",  # deep-orange
}
SUSPICIOUS_COLOR = "#FF1744"  # bright red for highlighted suspicious nodes

# Draws the circuit as a directed graph. 
# Suspicious nodes are outlined in bright red and enlarged; other nodes are coloured by gate type.

def visualizeCircuit(node_df: pd.DataFrame,
                      edge_df: pd.DataFrame,
                      suspicious_ids: list[int],
                      module_name: str,
                      output_path: str) -> None:

    G = nx.DiGraph()
    for _, row in node_df.iterrows():
        G.add_node(int(row["node_id"]),
                   label=row["signal_name"],
                   gtype=row["type"])

    valid_ids = set(node_df["node_id"].tolist())
    for _, row in edge_df.iterrows():
        s, d = int(row["src"]), int(row["dst"])
        if s in valid_ids and d in valid_ids:
            G.add_edge(s, d)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            pos = nx.planar_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=1.5)

    susp_set = set(suspicious_ids)
    node_list  = list(G.nodes())
    colors     = [NODE_COLORS.get(G.nodes[n].get("gtype","assign"), "#999999")
                  for n in node_list]
    sizes      = [1200 if n in susp_set else 600 for n in node_list]
    edgecolors = [SUSPICIOUS_COLOR if n in susp_set else "#333333"
                  for n in node_list]
    linewidths = [3.0 if n in susp_set else 0.8 for n in node_list]
    labels     = {n: G.nodes[n].get("label", str(n)) for n in node_list}

    fig, ax = plt.subplots(figsize=(max(14, len(node_list) * 0.4),
                                    max(10, len(node_list) * 0.25)))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4,
                           edge_color="#555555", arrowsize=15,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=colors,
                           node_size=sizes, edgecolors=edgecolors,
                           linewidths=linewidths, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7,
                            font_weight="bold", ax=ax)

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, label=t.upper())
        for t, c in NODE_COLORS.items()
        if any(G.nodes[n].get("gtype") == t for n in G.nodes)
    ]
    legend_handles.append(
        mpatches.Patch(edgecolor=SUSPICIOUS_COLOR, facecolor="white",
                       linewidth=2, label="⚠ Suspicious"))
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=8, framealpha=0.8)

    ax.set_title(f"Circuit Graph — module {module_name}\n"
                 f"(red outlines = suspicious nodes)", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# =============================================================================

# Core Analysis Pipeline

def analyze(verilog_path: str,
            model: FaultGNN,
            top_k: int = 10,
            save_graph: bool = True) -> dict:
    """
    Run the full fault-analysis pipeline on a single Verilog file.

    Returns a structured report dict.
    """
    stem = Path(verilog_path).stem

    # ── Parse ────────────────────────────────────────────────────────────────
    node_df, edge_df, meta = parseVerilog(verilog_path)

    if node_df.empty:
        return {"error": "Parser returned no nodes.", "file": verilog_path}
    if edge_df.empty:
        return {"error": "Parser returned no edges — circuit may be empty or "
                         "use only behavioural constructs not supported by the "
                         "structural parser.", "file": verilog_path}

    # ── Build PyG graph ───────────────────────────────────────────────────────
    data = build_PYGdata(node_df, edge_df, meta)
    if data is None:
        return {"error": "Could not build a valid graph (no edges after filtering).",
                "file": verilog_path}

    data = data.to(DEVICE)

    # ── Graph-level prediction ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits     = model(data)
        probs      = F.softmax(logits, dim=1)[0]
        fault_prob = float(probs[1].item())
        healthy_prob = float(probs[0].item())
        is_faulty  = bool(probs.argmax().item() == 1)

    # ── Node suspicion scores ─────────────────────────────────────────────────
    scores  = model.node_fault_scores(data)
    top_k   = min(top_k, len(node_df))
    top_idx = scores.argsort(descending=True)[:top_k].tolist()

    suspicious = []
    for rank, nidx in enumerate(top_idx, 1):
        rows = node_df[node_df["node_id"] == nidx]
        if rows.empty:
            continue
        row   = rows.iloc[0]
        gtype = str(row["type"]).lower()
        suspicious.append({
            "rank":         rank,
            "node_id":      int(nidx),
            "signal_name":  str(row["signal_name"]),
            "gate_type":    gtype,
            "line_no":      int(row["line_no"]),
            "fanin":        int(row["fanin"]),
            "fanout":       int(row["fanout"]),
            "score":        round(float(scores[nidx].item()), 4),
            "suggestions":  FAULT_SUGGESTIONS.get(gtype,
                                                   FAULT_SUGGESTIONS["default"]),
        })

    # ── Save graph visualisation ──────────────────────────────────────────────
    graph_path = None
    if save_graph:
        graph_path = f"{stem}_graph.png"
        susp_ids = [n["node_id"] for n in suspicious]
        visualizeCircuit(node_df, edge_df, susp_ids,
                          meta["module_name"], graph_path)

    return {
        "file":              verilog_path,
        "module_name":       meta["module_name"],
        "num_signals":       meta["num_signals"],
        "num_gates":         meta["num_gates"],
        "num_inputs":        meta["num_inputs"],
        "num_outputs":       meta["num_outputs"],
        "num_edges":         len(edge_df),
        "is_faulty":         is_faulty,
        "fault_probability": round(fault_prob, 4),
        "healthy_probability": round(healthy_prob, 4),
        "suspicious_nodes":  suspicious,
        "graph_image":       graph_path,
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
    }

# =============================================================================

# Formatting for terminal output

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def _bar(prob: float, width: int = 30) -> str:
    filled = round(prob * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {prob*100:5.1f}%"


def print_report(report: dict, verbose: bool = True) -> None:
    """Pretty-print a single analysis report to stdout."""
    if "error" in report:
        print(f"\n{RED}✗ Error:{RESET} {report['error']}")
        print(f"  File: {report.get('file','')}")
        return

    verdict = (f"{RED}{BOLD}⚠  FAULTY{RESET}"
               if report["is_faulty"] else f"{GREEN}{BOLD}✓  HEALTHY{RESET}")

    print(f"\n{'═'*65}")
    print(f"  {BOLD}File     :{RESET} {report['file']}")
    print(f"  {BOLD}Module   :{RESET} {report['module_name']}")
    print(f"  {BOLD}Signals  :{RESET} {report['num_signals']}   "
          f"Gates: {report['num_gates']}   "
          f"Edges: {report['num_edges']}")
    print(f"  {BOLD}Ports    :{RESET} "
          f"in={report['num_inputs']}  out={report['num_outputs']}")
    print(f"{'─'*65}")
    print(f"\n  {BOLD}Verdict  :{RESET} {verdict}")
    print(f"\n  {BOLD}Fault probability{RESET}")
    print(f"    Faulty  {RED}{_bar(report['fault_probability'])}{RESET}")
    print(f"    Healthy {GREEN}{_bar(report['healthy_probability'])}{RESET}")

    susp = report.get("suspicious_nodes", [])
    if susp:
        top_k = len(susp)
        print(f"\n{'─'*65}")
        print(f"  {BOLD}Top-{top_k} Suspicious Nodes{RESET}")
        print(f"\n  {'#':<4} {'Signal':<22} {'Type':<8} "
              f"{'Line':<6} {'Fan-in':<8} {'Fan-out':<9} {'Score'}")
        print(f"  {'─'*60}")
        for n in susp:
            bar_len  = max(1, round(n["score"] * 15))
            score_bar = f"{'█'*bar_len}{'░'*(15-bar_len)} {n['score']:.3f}"
            color    = RED if n["rank"] <= 3 else (YELLOW if n["rank"] <= 6
                                                   else RESET)
            ln = str(n["line_no"]) if n["line_no"] > 0 else "—"
            print(f"  {color}{n['rank']:<4} {n['signal_name']:<22} "
                  f"{n['gate_type']:<8} {ln:<6} {n['fanin']:<8} "
                  f"{n['fanout']:<9} {score_bar}{RESET}")

    if verbose and susp:
        print(f"\n{'─'*65}")
        print(f"  {BOLD}Fix Suggestions{RESET}")
        for n in susp:
            ln = f" (line {n['line_no']})" if n["line_no"] > 0 else ""
            color = RED if n["rank"] <= 3 else (YELLOW if n["rank"] <= 6
                                                else RESET)
            print(f"\n  {color}[{n['rank']}] {n['signal_name']}{ln}"
                  f"  ‹{n['gate_type']}›{RESET}")
            for hint in n["suggestions"]:
                wrapped = textwrap.fill(hint, width=60,
                                        initial_indent="       • ",
                                        subsequent_indent="         ")
                print(wrapped)

    if report.get("graph_image"):
        print(f"\n  {DIM}Graph saved  → {report['graph_image']}{RESET}")
    print(f"\n{'═'*65}")

# =============================================================================

# Interactive Prompt Mode

def interactive_mode():
    """
    Walk the user through model path → Verilog file selection → analysis loop.
    Keeps running until the user types 'quit'.
    """
    print(f"\n{BOLD}{'═'*65}")
    print("  NeuroChip Smart Fault Analyzer — Interactive Mode")
    print(f"{'═'*65}{RESET}")
    print(f"  Device: {DEVICE}")
    default_model_path = str(Path(__file__).resolve().with_name("NeuroChip.pt"))
    print(f"  Default model: {default_model_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    while True:
        model_path = input(
            f"\n{BOLD}Enter path to trained model (.pt file)"
            f" [default: {default_model_path}]:{RESET} "
        ).strip().strip('"').strip("'")
        try:
            model = load_model(model_path or None)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n  {GREEN}✓ Model loaded{RESET}  "
                  f"({n_params:,} parameters, device={DEVICE})")
            break
        except FileNotFoundError as e:
            print(f"  {RED}✗ {e}{RESET}")
        except Exception as e:
            print(f"  {RED}✗ Failed to load model: {e}{RESET}")

    # ── Ask for settings ──────────────────────────────────────────────────────
    while True:
        try:
            top_k = int(input(
                f"\n{BOLD}How many suspicious nodes to report? [default 10]:{RESET} "
            ).strip() or "10")
            if top_k < 1:
                raise ValueError
            break
        except ValueError:
            print("  Please enter a positive integer.")

    save_json_path = input(
        f"\n{BOLD}Save full JSON report to file? "
        f"(leave blank to skip):{RESET} "
    ).strip().strip('"').strip("'") or None

    all_reports = []

    # ── Analysis loop ─────────────────────────────────────────────────────────
    print(f"\n  {DIM}Type the path to a Verilog file to analyse it.")
    print(f"  Type 'quit' or press Ctrl-C to exit.{RESET}")

    while True:
        try:
            raw = input(f"\n{BOLD}Verilog file path (or 'quit'):{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            raw = "quit"

        if raw.lower() in {"quit", "exit", "q", ""}:
            break

        verilog_path = raw.strip('"').strip("'")
        if not verilog_path.endswith(".v") and not verilog_path.endswith(".sv"):
            print(f"  {YELLOW}Warning: file does not end in .v / .sv "
                  f"— proceeding anyway.{RESET}")

        print(f"\n  {DIM}Analysing…{RESET}", end="", flush=True)
        report = analyze(verilog_path, model, top_k=top_k)
        print(f"\r            \r", end="")
        print_report(report, verbose=True)
        all_reports.append(report)

    # ── Save consolidated JSON ────────────────────────────────────────────────
    if save_json_path and all_reports:
        with open(save_json_path, "w") as fj:
            json.dump(all_reports, fj, indent=2)
        print(f"\n  {GREEN}✓ Report saved → {save_json_path}{RESET}")

    print(f"\n  {BOLD}Session complete.{RESET}  "
          f"Analysed {len(all_reports)} file(s).\n")

# =============================================================================

# CLI Entry Point

def main():
    parser = argparse.ArgumentParser(
        prog="neurochip_analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            NeuroChip Smart Fault Analyzer
            ──────────────────────────────
            GNN-powered fault detection and localisation for Verilog designs.

            Run without arguments for interactive guided mode.
        """),
    )
    parser.add_argument("--model", "-m", metavar="MODEL.pt",
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--verilog", "-v", metavar="FILE.v", nargs="+",
                        help="One or more Verilog source files to analyse")
    parser.add_argument("--top-k", "-k", type=int, default=10,
                        help="Number of suspicious nodes to report (default: 10)")
    parser.add_argument("--save-report", "-r", metavar="REPORT.json",
                        help="Save full JSON report to this file")
    parser.add_argument("--no-graph", action="store_true",
                        help="Skip saving the graph visualisation PNG")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress per-node fix suggestions in output")
    args = parser.parse_args()

    # If no CLI arguments, switch to interactive mode
    if not args.model and not args.verilog:
        interactive_mode()
        return

    # ── Validate CLI inputs ───────────────────────────────────────────────────
    if not args.verilog:
        parser.error("--verilog is required when running in CLI mode.")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = args.model or str(Path(__file__).resolve().with_name("NeuroChip.pt"))
    print(f"\n  Loading model from {model_path} …", end="", flush=True)
    try:
        model = load_model(args.model)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\r  {GREEN}✓ Model loaded{RESET}  "
              f"({n_params:,} params, device={DEVICE})")
    except Exception as e:
        print(f"\n{RED}✗ Could not load model: {e}{RESET}")
        sys.exit(1)

    # ── Analyse each file ─────────────────────────────────────────────────────
    all_reports = []
    for vf in args.verilog:
        report = analyze(
            vf, model,
            top_k       = args.top_k,
            save_graph  = not args.no_graph,
        )
        print_report(report, verbose=not args.quiet)
        all_reports.append(report)

    # ── Save JSON report ──────────────────────────────────────────────────────
    if args.save_report:
        with open(args.save_report, "w") as fj:
            json.dump(all_reports, fj, indent=2)
        print(f"\n  {GREEN}✓ JSON report saved → {args.save_report}{RESET}\n")

if __name__ == "__main__":
    main()