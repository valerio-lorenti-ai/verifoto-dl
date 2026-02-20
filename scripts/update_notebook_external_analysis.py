#!/usr/bin/env python3
"""
Script to add external test photo-level analysis cell to the notebook.
"""
import json
import sys

def add_external_analysis_cell(notebook_path):
    """Add photo-level analysis cell for external test."""
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # New cell code
    new_cell_code = [
        "# ============================================================================\n",
        "# EXTERNAL TEST: PHOTO-LEVEL ANALYSIS & CALIBRATION\n",
        "# ============================================================================\n",
        "\n",
        "EXTERNAL_RUN_NAME = f\"{EXPERIMENT_NAME}_external\"\n",
        "EXTERNAL_OUTPUT_DIR = f\"outputs/runs/{EXTERNAL_RUN_NAME}\"\n",
        "\n",
        "if os.path.exists(f\"{EXTERNAL_OUTPUT_DIR}/metrics.json\"):\n",
        "    print(\"=\"*80)\n",
        "    print(\"EXTERNAL TEST: PHOTO-LEVEL ANALYSIS\")\n",
        "    print(\"=\"*80)\n",
        "    \n",
        "    # Photo-level analysis\n",
        "    !python scripts/analyze_by_photo.py \\\n",
        "        --run {EXTERNAL_OUTPUT_DIR} \\\n",
        "        --min-recall 0.90\n",
        "    \n",
        "    # Display chosen threshold\n",
        "    if os.path.exists(f\"{EXTERNAL_OUTPUT_DIR}/chosen_threshold.json\"):\n",
        "        import json\n",
        "        with open(f\"{EXTERNAL_OUTPUT_DIR}/chosen_threshold.json\", 'r') as f:\n",
        "            threshold_info = json.load(f)\n",
        "        \n",
        "        print(\"\\n\" + \"=\"*80)\n",
        "        print(\"EXTERNAL TEST: THRESHOLD RECOMMENDATION\")\n",
        "        print(\"=\"*80)\n",
        "        print(f\"🎯 Recommended threshold: {threshold_info['recommendation']:.3f}\")\n",
        "        print(f\"   F1-optimal threshold: {threshold_info['f1_optimal']:.3f}\")\n",
        "        print(f\"   Precision-optimal (recall≥90%): {threshold_info['precision_optimal_recall_90']:.3f}\")\n",
        "        print(f\"\\nRationale: {threshold_info['rationale']}\")\n",
        "        \n",
        "        # Display photo-level metrics\n",
        "        with open(f\"{EXTERNAL_OUTPUT_DIR}/photo_level_metrics.json\", 'r') as f:\n",
        "            photo_metrics = json.load(f)\n",
        "        \n",
        "        print(\"\\n\" + \"=\"*80)\n",
        "        print(\"EXTERNAL TEST: PHOTO-LEVEL METRICS\")\n",
        "        print(\"=\"*80)\n",
        "        print(f\"Total photos: {photo_metrics['n_photos']}\")\n",
        "        \n",
        "        print(\"\\nOriginal threshold (from training):\")\n",
        "        orig = photo_metrics['original_threshold']\n",
        "        print(f\"  Threshold: {orig['threshold']:.3f}\")\n",
        "        print(f\"  Precision: {orig['precision']:.1%}\")\n",
        "        print(f\"  Recall:    {orig['recall']:.1%}\")\n",
        "        print(f\"  F1:        {orig['f1']:.1%}\")\n",
        "        print(f\"  FP: {orig['fp']}, FN: {orig['fn']}\")\n",
        "        \n",
        "        print(f\"\\nOptimal threshold for external ({threshold_info['recommendation']:.3f}):\")\n",
        "        opt = photo_metrics['optimal_f1_threshold']\n",
        "        print(f\"  Precision: {opt['precision']:.1%} ({opt['precision']-orig['precision']:+.1%})\")\n",
        "        print(f\"  Recall:    {opt['recall']:.1%} ({opt['recall']-orig['recall']:+.1%})\")\n",
        "        print(f\"  F1:        {opt['f1']:.1%} ({opt['f1']-orig['f1']:+.1%})\")\n",
        "        print(f\"  FP: {opt['fp']} ({int(opt['fp']-orig['fp']):+d}), FN: {opt['fn']} ({int(opt['fn']-orig['fn']):+d})\")\n",
        "        \n",
        "        print(\"\\n✓ External photo-level analysis complete\")\n",
        "    \n",
        "    # Temperature scaling calibration (copy from training)\n",
        "    print(\"\\n\" + \"=\"*80)\n",
        "    print(\"EXTERNAL TEST: CALIBRATION\")\n",
        "    print(\"=\"*80)\n",
        "    print(\"⚠️  Note: External test uses model calibrated on internal validation data\")\n",
        "    print(\"    Calibration parameters from training run are already applied.\")\n",
        "    \n",
        "    # Copy calibration files from training run\n",
        "    import shutil\n",
        "    training_cal_T = f\"{OUTPUT_DIR}/calibration_T.json\"\n",
        "    training_cal_report = f\"{OUTPUT_DIR}/calibration_report.json\"\n",
        "    \n",
        "    if os.path.exists(training_cal_T):\n",
        "        shutil.copy(training_cal_T, f\"{EXTERNAL_OUTPUT_DIR}/calibration_T.json\")\n",
        "        print(f\"✓ Copied calibration_T.json from training run\")\n",
        "    \n",
        "    if os.path.exists(training_cal_report):\n",
        "        shutil.copy(training_cal_report, f\"{EXTERNAL_OUTPUT_DIR}/calibration_report.json\")\n",
        "        print(f\"✓ Copied calibration_report.json from training run\")\n",
        "    \n",
        "    print(\"\\n✓ External test analysis complete\")\n",
        "    print(\"=\"*80)\n",
        "else:\n",
        "    print(\"\\n⚠️  External test not found - skipping analysis\")"
    ]
    
    # Create new cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_cell_code
    }
    
    # Find the position to insert (after external test, before git push)
    # Look for the cell that contains "PUSH RESULTS TO GITHUB"
    insert_position = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'PUSH RESULTS TO GITHUB' in source:
                insert_position = i
                break
    
    if insert_position is None:
        print("❌ Could not find 'PUSH RESULTS TO GITHUB' section")
        return False
    
    # Insert the new cell
    notebook['cells'].insert(insert_position, new_cell)
    
    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Added external analysis cell at position {insert_position}")
    print(f"✓ Notebook updated: {notebook_path}")
    return True

if __name__ == '__main__':
    notebook_path = 'scripts/notebooks/verifoto_dl.ipynb'
    success = add_external_analysis_cell(notebook_path)
    sys.exit(0 if success else 1)
