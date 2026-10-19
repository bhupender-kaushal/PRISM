import os
import json
from pathlib import Path
import pandas as pd

# ==========================================
# DIAGNOSTIC CONFIGURATION
# ==========================================
CONT_DIR = Path("/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/prism_out_original")
ID_DIR = Path("/bhupender/2025/Mouse_other/Skin_Lesion/TransFuse/prism_out_insertion_deletion")

def find_id_trajectories(node):
    if isinstance(node, dict):
        if 'trajectories' in node and isinstance(node['trajectories'], dict):
            return node['trajectories']
        for v in node.values():
            res = find_id_trajectories(v)
            if res: return res
    return {}

print("==================================================")
print("🔍 TRANSFUSE DIAGNOSTIC TOOL")
print("==================================================")

# --------------------------------------------------
# 1. CHECK CONTINUOUS DIRECTORY
# --------------------------------------------------
print("\n[1] Checking Continuous Data (S-Factor / Global AUC)...")
if not CONT_DIR.exists():
    print(f"❌ ERROR: Continuous Directory does not exist:\n   {CONT_DIR}")
else:
    run_1_path = CONT_DIR / "run_1"
    if not run_1_path.exists():
        print(f"❌ ERROR: 'run_1' folder missing inside Continuous Directory:\n   {run_1_path}")
    else:
        cont_files = list(run_1_path.rglob('*_analysis.json'))
        print(f"✅ Found {len(cont_files)} JSON files in Continuous directory.")
        
        if len(cont_files) > 0:
            test_file = cont_files[0]
            try:
                with open(test_file, 'r') as f: data = json.load(f)
                keys = list(data.keys())
                print(f"   -> Root keys in first file: {keys}")
                if 'in_context_analysis' not in keys: print("   ❌ Missing 'in_context_analysis'")
                if 'out_of_context_analysis' not in keys: print("   ❌ Missing 'out_of_context_analysis'")
            except Exception as e:
                print(f"   ❌ Failed to load JSON: {e}")

# --------------------------------------------------
# 2. CHECK I/D DIRECTORY
# --------------------------------------------------
print("\n[2] Checking I/D Data (Deletion / Insertion)...")
id_records = []
if not ID_DIR.exists():
    print(f"❌ ERROR: I/D Directory does not exist:\n   {ID_DIR}")
else:
    id_files = list(ID_DIR.rglob('*_analysis.json'))
    print(f"✅ Found {len(id_files)} JSON files in I/D directory.")
    
    success_count = 0
    missing_prob_ranked = 0
    missing_baselines = 0
    
    for p in id_files:
        image_id = p.stem.replace('_analysis', '')
        try:
            with open(p, 'r') as f: data = json.load(f)
            trajectories = find_id_trajectories(data)
            
            if not trajectories:
                print(f"   ❌ [Image {image_id}] 'trajectories' dict completely missing/empty.")
                continue
                
            has_valid_baseline = False
            for baseline in ['black', 'mean']:
                if baseline in trajectories:
                    has_valid_baseline = True
                    ranks = trajectories[baseline]
                    if 'prob_ranked' in ranks:
                        id_records.append({'image_id': image_id, 'baseline': baseline})
                    else:
                        missing_prob_ranked += 1
            
            if has_valid_baseline: success_count += 1
            else: missing_baselines += 1
                
        except Exception as e: pass

    print(f"   -> Successfully extracted I/D baselines for {success_count} images.")
    if missing_baselines > 0:
        print(f"   -> ⚠️ {missing_baselines} images had NO 'black' or 'mean' baselines.")
    if missing_prob_ranked > 0:
        print(f"   -> ❌ 'prob_ranked' was missing in {missing_prob_ranked} baseline entries!")

# --------------------------------------------------
# 3. PIVOT TABLE / OVERLAP SIMULATION (I/D)
# --------------------------------------------------
print("\n[3] Overlap Check (Can we form a grid for Kendall's W?)...")
if len(id_records) > 0:
    df = pd.DataFrame(id_records)
    # We add a dummy score just to test the pivot table mechanics
    df['score'] = 1.0 
    
    try:
        pivot = df.pivot_table(index='image_id', columns='baseline', values='score')
        initial_images = len(pivot)
        pivot_clean = pivot.dropna(axis=0, how='any')
        surviving_images = len(pivot_clean)
        
        print(f"   -> Total unique images found: {initial_images}")
        print(f"   -> Images with BOTH 'black' and 'mean' baselines valid: {surviving_images}")
        
        if surviving_images < 20:
            print(f"\n🚨 CAUSE IDENTIFIED: You need 20 perfectly overlapping images to run the stats, but TransFuse only has {surviving_images}.")
        else:
            print("\n✅ TransFuse I/D data looks perfectly healthy. If it is failing, the issue is with the Continuous data.")
    except Exception as e:
        print(f"   ❌ Pivot table simulation failed: {e}")
else:
    print("   ❌ No I/D records extracted. Cannot test overlap.")
print("==================================================")