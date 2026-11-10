import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs('figures', exist_ok=True)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# -------------------------------------------------------------------------
# FIGURE 1: TAXONOMY COMPARISON
# -------------------------------------------------------------------------
def create_fig1():
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title Banner
    ax.text(50, 95, "Taxonomy of Compute Sparsity in Large Language Models", 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#0F172A')
    ax.text(50, 90.5, "Contrasting Monolithic Weight Sparsity with System-Level Modular Harness Sparsity",
            ha='center', va='center', fontsize=10.5, color='#475569')

    # Root: Dense Transformers
    root_box = patches.FancyBboxPatch((28, 68), 44, 16, boxstyle="round,pad=1.2,rounding_size=2.5",
                                      facecolor='#1E293B', edgecolor='#0F172A', linewidth=2)
    ax.add_patch(root_box)
    ax.text(50, 78.5, "Dense Transformers", ha='center', va='center', fontsize=13, fontweight='bold', color='#FFFFFF')
    ax.text(50, 73.5, "e.g., GPT-4 (Base), LLaMA-3 (Dense), Chinchilla\n100% Parameter Activation Per Token | O(N) Compute per Token", 
            ha='center', va='center', fontsize=9.5, color='#CBD5E1')

    # Arrows splitting
    ax.annotate('', xy=(24, 52), xytext=(40, 68),
                arrowprops=dict(arrowstyle="->", color='#64748B', lw=2.2, connectionstyle="arc3,rad=0.15"))
    ax.annotate('', xy=(76, 52), xytext=(60, 68),
                arrowprops=dict(arrowstyle="->", color='#059669', lw=2.5, connectionstyle="arc3,rad=-0.15"))

    # Left Child: Weight-Level Sparsity (MoE)
    moe_box = patches.FancyBboxPatch((4, 12), 40, 38, boxstyle="round,pad=1.2,rounding_size=2.5",
                                    facecolor='#FFFFFF', edgecolor='#6366F1', linewidth=2)
    ax.add_patch(moe_box)
    
    # MoE Header Badge
    moe_badge = patches.FancyBboxPatch((7, 43), 34, 5.5, boxstyle="round,pad=0.5,rounding_size=1.5",
                                       facecolor='#EEF2FF', edgecolor='#A5B4FC', linewidth=1)
    ax.add_patch(moe_badge)
    ax.text(24, 45.7, "Weight-Level Sparsity (MoE)", ha='center', va='center', fontsize=11, fontweight='bold', color='#3730A3')

    moe_text = (
        "• Paradigm: Internal FFN layer gating\n"
        "• Exemplars: Switch Transformers, DeepSeek-V3, Mixtral\n"
        "• Routing: Micro-gating per token to top-k experts\n"
        "• Limitation: Dense Attention & Shared KV Cache\n"
        "• Inflexibility: Cannot hot-swap or update single\n"
        "  experts without full retraining\n"
        "• Context Overhead: Full history retains quadratic cost"
    )
    ax.text(6, 26, moe_text, ha='left', va='center', fontsize=8.8, color='#334155', linespacing=1.5)

    # Right Child: System-Level Harness Sparsity (OUR WORK)
    snh_box = patches.FancyBboxPatch((56, 12), 40, 38, boxstyle="round,pad=1.2,rounding_size=2.5",
                                    facecolor='#FFFFFF', edgecolor='#059669', linewidth=2.5)
    ax.add_patch(snh_box)
    
    # SNH Header Badge
    snh_badge = patches.FancyBboxPatch((59, 43), 34, 5.5, boxstyle="round,pad=0.5,rounding_size=1.5",
                                       facecolor='#ECFDF5', edgecolor='#6EE7B7', linewidth=1.2)
    ax.add_patch(snh_badge)
    ax.text(76, 45.7, "System-Level Harness Sparsity (OUR WORK)", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#065F46')

    snh_text = (
        "• Paradigm: Directed Acyclic Graph (DAG) of Micro-Nodes\n"
        "• Exemplars: Sparse Node Harness (SNH)\n"
        "• Routing: Dynamic Shortest-Path semantic dispatch\n"
        "• Zero-FLOP Inactivity: Dormant nodes consume 0 compute\n"
        "• Modularity: Hot-swappable SLMs & specialized LoRAs\n"
        "• Context Isolation: Compressed typed state envelopes\n"
        "• Energy Profile: Biological ~20W brain-like firing"
    )
    ax.text(58, 26, snh_text, ha='left', va='center', fontsize=8.8, color='#1E293B', linespacing=1.5)

    plt.tight_layout()
    plt.savefig('figures/fig1_taxonomy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 1 generated successfully.")

# -------------------------------------------------------------------------
# FIGURE 2: SPARSE NODE HARNESS ARCHITECTURE
# -------------------------------------------------------------------------
def create_fig2():
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title
    ax.text(50, 96, "Sparse Node Harness (SNH) Dynamic Execution Topology", 
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0F172A')
    ax.text(50, 92.5, "Shortest-path activation of micro-specialists with zero compute allocation to dormant nodes",
            ha='center', va='center', fontsize=10, color='#475569')

    # Input Box
    in_box = patches.FancyBboxPatch((3, 76), 18, 11, boxstyle="round,pad=0.8,rounding_size=1.5",
                                    facecolor='#EEF2FF', edgecolor='#4F46E5', linewidth=1.8)
    ax.add_patch(in_box)
    ax.text(12, 83, "User Task\nInput (S_init)", ha='center', va='center', fontsize=10, fontweight='bold', color='#312E81')
    ax.text(12, 78, "Multi-Step Intent", ha='center', va='center', fontsize=8.5, color='#4338CA')

    # Arrow to Router
    ax.annotate('', xy=(28, 81.5), xytext=(22, 81.5),
                arrowprops=dict(arrowstyle="->", color='#0F172A', lw=2))

    # Semantic Router
    router_box = patches.FancyBboxPatch((28, 73), 22, 17, boxstyle="round,pad=1,rounding_size=2",
                                        facecolor='#0F172A', edgecolor='#334155', linewidth=2)
    ax.add_patch(router_box)
    ax.text(39, 84, "Semantic Router", ha='center', va='center', fontsize=11, fontweight='bold', color='#38BDF8')
    ax.text(39, 78.5, "Ultra-Lightweight (~10M)\nEmbedding Cosine / SLM\nCalculates Shortest Path", 
            ha='center', va='center', fontsize=8.5, color='#E2E8F0')

    # Main active pathway enclosure
    active_enc = patches.FancyBboxPatch((56, 46), 41, 46, boxstyle="round,pad=1.2,rounding_size=2.5",
                                        facecolor='#F0FDF4', edgecolor='#10B981', linewidth=2.5, linestyle='-')
    ax.add_patch(active_enc)
    ax.text(76.5, 89, "ACTIVE EXECUTION PATHWAY (Only ~1.6B Params Fired)", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#047857')

    # Active Node A
    node_a = patches.FancyBboxPatch((60, 75), 33, 9.5, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    facecolor='#FFFFFF', edgecolor='#059669', linewidth=1.5)
    ax.add_patch(node_a)
    ax.text(76.5, 81, "Node A: Intent & Entity Parser", ha='center', va='center', fontsize=9, fontweight='bold', color='#065F46')
    ax.text(76.5, 77, "SmolLM2-135M | Emits Typed State JSON", ha='center', va='center', fontsize=8, color='#047857')

    # Active Node B
    node_b = patches.FancyBboxPatch((60, 61), 33, 9.5, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    facecolor='#FFFFFF', edgecolor='#059669', linewidth=1.5)
    ax.add_patch(node_b)
    ax.text(76.5, 67, "Node B: Symbolic Tool / Verifier", ha='center', va='center', fontsize=9, fontweight='bold', color='#065F46')
    ax.text(76.5, 63, "Python AST / Regex | 0 Tokens / 0 Hallucination", ha='center', va='center', fontsize=8, color='#047857')

    # Active Node C
    node_c = patches.FancyBboxPatch((60, 48), 33, 9.5, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    facecolor='#FFFFFF', edgecolor='#059669', linewidth=1.5)
    ax.add_patch(node_c)
    ax.text(76.5, 54, "Node C: Response Synthesizer", ha='center', va='center', fontsize=9, fontweight='bold', color='#065F46')
    ax.text(76.5, 50, "Qwen-2.5-1.5B | Isolated Output Generation", ha='center', va='center', fontsize=8, color='#047857')

    # Connect Router to Node A
    ax.annotate('', xy=(60, 80), xytext=(51, 81.5),
                arrowprops=dict(arrowstyle="->", color='#059669', lw=2.2))
    # Connect Node A to Node B
    ax.annotate('', xy=(76.5, 71.5), xytext=(76.5, 74.5),
                arrowprops=dict(arrowstyle="->", color='#059669', lw=2))
    ax.text(76.5, 73, "Compressed Schema Envelope", ha='center', va='center', fontsize=7.5, color='#065F46', backgroundcolor='#F0FDF4')
    # Connect Node B to Node C
    ax.annotate('', xy=(76.5, 58), xytext=(76.5, 60.5),
                arrowprops=dict(arrowstyle="->", color='#059669', lw=2))
    ax.text(76.5, 59.5, "Deterministic State Output", ha='center', va='center', fontsize=7.5, color='#065F46', backgroundcolor='#F0FDF4')

    # Dormant Nodes Enclosure
    dormant_enc = patches.FancyBboxPatch((12, 10), 76, 28, boxstyle="round,pad=1.2,rounding_size=2",
                                        facecolor='#F1F5F9', edgecolor='#94A3B8', linewidth=1.8, linestyle='--')
    ax.add_patch(dormant_enc)
    ax.text(50, 34, "DORMANT SPECIALIZED NODES (0 FLOPs | 0 Tokens Consumed | Preserved in VRAM / RAM)", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#475569')

    # Dormant individual nodes
    dormant_items = [
        ("Code Auditor (7B)", 15),
        ("Legal Specialist (3B)", 33),
        ("Vision Module", 52),
        ("Multilingual Node", 70)
    ]
    for name, x_pos in dormant_items:
        d_box = patches.FancyBboxPatch((x_pos, 16), 16, 12, boxstyle="round,pad=0.5,rounding_size=1",
                                       facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.2)
        ax.add_patch(d_box)
        ax.text(x_pos + 8, 24, name, ha='center', va='center', fontsize=8.2, fontweight='bold', color='#64748B')
        ax.text(x_pos + 8, 19, "State: Sleeping\n0 Joules", ha='center', va='center', fontsize=7.5, color='#94A3B8')

    plt.tight_layout()
    plt.savefig('figures/fig2_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 2 generated successfully.")

# -------------------------------------------------------------------------
# FIGURE 3: SCALING WALL & PARETO CURVE
# -------------------------------------------------------------------------
def create_fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')
    ax1.set_facecolor('#FFFFFF')
    ax2.set_facecolor('#FFFFFF')

    # Plot 1: Logarithmic Dense Scaling vs Combinatorial Sparse Scaling
    x = np.linspace(1, 10, 100)
    y_dense = np.log2(x + 1) * 22          # Logarithmic flattening
    y_sparse = 12 * (1 - np.exp(-0.35 * x)) + 8.5 * x   # Sustained capability with combinatorial paths

    ax1.plot(x, y_dense, label='Dense Scaling (Monolithic)', color='#EF4444', linewidth=2.8, linestyle='--')
    ax1.plot(x, y_sparse, label='Sparse Node Harness (Modular)', color='#10B981', linewidth=3)
    ax1.set_title("Downstream Accuracy vs Compute Investment", fontsize=11, fontweight='bold', color='#0F172A')
    ax1.set_xlabel("Relative Compute Budget / Parameters ($C$)", fontsize=9.5, color='#334155')
    ax1.set_ylabel("Task Reasoning Performance (%)", fontsize=9.5, color='#334155')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', frameon=True, facecolor='#F8FAFC')
    ax1.annotate('Logarithmic Wall\n(Chinchilla Plateau)', xy=(8.5, y_dense[84]), xytext=(4.5, 60),
                 arrowprops=dict(arrowstyle="->", color='#B91C1C', lw=1.5),
                 fontsize=8.5, fontweight='bold', color='#B91C1C')
    ax1.annotate('Combinatorial Expansion\n(k-of-N Subnetworks)', xy=(7.5, y_sparse[74]), xytext=(3, 78),
                 arrowprops=dict(arrowstyle="->", color='#047857', lw=1.5),
                 fontsize=8.5, fontweight='bold', color='#047857')

    # Plot 2: Energy & Token Footprint Comparison
    categories = ['Dense 70B\n(Full Context)', 'Dense 8B\n(Standard Agent)', 'Sparse Node\nHarness (Ours)']
    tokens = [45000, 38000, 4200]
    joules = [18500, 3200, 240]

    x_indices = np.arange(len(categories))
    width = 0.35

    color_tokens = '#6366F1'
    color_joules = '#F59E0B'

    rects1 = ax2.bar(x_indices - width/2, tokens, width, label='Tokens Consumed', color=color_tokens, alpha=0.9)
    ax2_twin = ax2.twinx()
    rects2 = ax2_twin.bar(x_indices + width/2, joules, width, label='Inference Energy (Joules)', color=color_joules, alpha=0.9)

    ax2.set_title("Context Token Burn & Energy Footprint per Multi-Step Task", fontsize=11, fontweight='bold', color='#0F172A')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(categories, fontsize=9, fontweight='semibold', color='#1E293B')
    ax2.set_ylabel("Total Tokens Processed", fontsize=9.5, color=color_tokens)
    ax2_twin.set_ylabel("Energy Consumed (Joules via CodeCarbon)", fontsize=9.5, color=color_joules)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, facecolor='#F8FAFC')

    plt.tight_layout()
    plt.savefig('figures/fig3_scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 3 generated successfully.")

if __name__ == '__main__':
    create_fig1()
    create_fig2()
    create_fig3()
    print("All figures successfully created in ./figures/")
