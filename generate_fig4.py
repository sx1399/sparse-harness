import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')
    ax1.set_facecolor('#F8FAFC')
    ax2.set_facecolor('#FFFFFF')
    
    # -------------------------------------------------------------
    # SUBPLOT 1: ARCHITECTURAL DIAGRAM OF ADAPTIVE SPECULATIVE HARNESS
    # -------------------------------------------------------------
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axis('off')
    
    ax1.text(50, 96, "Adaptive Speculative Verification Architecture", 
             ha='center', va='center', fontsize=12, fontweight='bold', color='#0F172A')
    ax1.text(50, 91.5, "Early-Exit Validation: Burning Tokens Only When Disambiguation is Required",
             ha='center', va='center', fontsize=8.5, color='#475569')

    # Executive Planner Node
    p_box = patches.FancyBboxPatch((25, 76), 50, 11, boxstyle="round,pad=0.8,rounding_size=1.5",
                                   facecolor='#0F172A', edgecolor='#334155', linewidth=1.5)
    ax1.add_patch(p_box)
    ax1.text(50, 83, "Executive Planner Node (~500M)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#38BDF8')
    ax1.text(50, 78.5, "Compact Schema Dispatch | Minimal Token Signature", ha='center', va='center', fontsize=8, color='#E2E8F0')

    # Arrow to Primary Worker
    ax1.annotate('', xy=(50, 65), xytext=(50, 76), arrowprops=dict(arrowstyle="->", color='#64748B', lw=1.8))

    # Primary Worker Box
    w1_box = patches.FancyBboxPatch((20, 53), 60, 12, boxstyle="round,pad=0.8,rounding_size=1.5",
                                    facecolor='#EEF2FF', edgecolor='#6366F1', linewidth=1.8)
    ax1.add_patch(w1_box)
    ax1.text(50, 61, "Primary Micro-Worker (135M)", ha='center', va='center', fontsize=9, fontweight='bold', color='#312E81')
    ax1.text(50, 56.5, "Initial Candidate Extraction (Compact Key-Value)", ha='center', va='center', fontsize=8, color='#4338CA')

    # Arrow to Deterministic Validator Gate
    ax1.annotate('', xy=(50, 41), xytext=(50, 53), arrowprops=dict(arrowstyle="->", color='#059669', lw=2))

    # Deterministic Gate (Diamond / Box)
    gate_box = patches.FancyBboxPatch((15, 27), 70, 14, boxstyle="round,pad=0.8,rounding_size=1.5",
                                      facecolor='#ECFDF5', edgecolor='#10B981', linewidth=2)
    ax1.add_patch(gate_box)
    ax1.text(50, 36.5, "Deterministic AST & Schema Validation Gate", ha='center', va='center', fontsize=9, fontweight='bold', color='#065F46')
    ax1.text(50, 31, "[PASS] -> Early Exit to Synthesizer (-66.1% Tokens, 75.2x Lower FLOPs)\n[FAIL] -> Triggers Speculative Branching Workers 2 & 3", 
             ha='center', va='center', fontsize=7.5, color='#047857')

    # Arrow from Gate to Synthesizer
    ax1.annotate('', xy=(50, 15), xytext=(50, 27), arrowprops=dict(arrowstyle="->", color='#0F172A', lw=2))
    ax1.text(52, 21, "Direct Early Exit", ha='left', va='center', fontsize=7.5, fontweight='bold', color='#059669')

    # Synthesizer
    s_box = patches.FancyBboxPatch((25, 4), 50, 10, boxstyle="round,pad=0.8,rounding_size=1.5",
                                   facecolor='#1E293B', edgecolor='#0F172A', linewidth=1.5)
    ax1.add_patch(s_box)
    ax1.text(50, 10.5, "Verified Synthesizer (500M)", ha='center', va='center', fontsize=9, fontweight='bold', color='#FFFFFF')
    ax1.text(50, 6.5, "Emits Final Answer (100% Exact Match)", ha='center', va='center', fontsize=7.5, color='#94A3B8')

    # -------------------------------------------------------------
    # SUBPLOT 2: EMPIRICAL PERFORMANCE ON COMPLEX WORKFLOWS
    # -------------------------------------------------------------
    categories = ['Dense 8B\n(Single-Pass ReAct)', 'Adaptive Sparse\nHarness (Ours)']
    accuracy = [40.0, 100.0]
    flops_g = [16.2, 0.215] # GigaFLOPs equivalent (x10^12)

    x_ind = np.arange(len(categories))
    width = 0.35

    color_acc = '#10B981'
    color_flops = '#F59E0B'

    rects1 = ax2.bar(x_ind - width/2, accuracy, width, label='Reasoning Accuracy (%)', color=color_acc, alpha=0.9)
    ax2_twin = ax2.twinx()
    rects2 = ax2_twin.bar(x_ind + width/2, flops_g, width, label='Inference FLOPs ($10^{12}$)', color=color_flops, alpha=0.9)

    ax2.set_title("Complex Reasoning: +60% Accuracy at 75.2x Lower Compute", fontsize=11, fontweight='bold', color='#0F172A')
    ax2.set_xticks(x_ind)
    ax2.set_xticklabels(categories, fontsize=9.5, fontweight='bold', color='#1E293B')
    ax2.set_ylabel("Task Reasoning Accuracy (%)", fontsize=9.5, color='#047857')
    ax2.set_ylim(0, 115)
    ax2_twin.set_ylabel("Inference Compute (FLOPs $\\times 10^{12}$)", fontsize=9.5, color=color_flops)
    ax2_twin.set_ylim(0, 20)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

    # Annotate accuracy boost
    ax2.annotate('+60.0% Accuracy Boost\n(100% Exact Match)', xy=(1 - width/2, 100), xytext=(0.45, 105),
                 arrowprops=dict(arrowstyle="->", color='#047857', lw=1.5),
                 fontsize=8.5, fontweight='bold', color='#047857')
    ax2_twin.annotate('75.2x Lower Compute\n(-66.1% Tokens: 342 vs 1010)', xy=(1 + width/2, 0.215), xytext=(0.45, 5),
                      arrowprops=dict(arrowstyle="->", color='#D97706', lw=1.5),
                      fontsize=8.5, fontweight='bold', color='#D97706')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='#F8FAFC')

    plt.tight_layout()
    plt.savefig('figures/fig4_complex_workflows.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 4 successfully updated with adaptive results.")

if __name__ == '__main__':
    create_fig4()
