import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from rdkit import RDLogger
import os


from generate_model import ScaffoldGenerator
from run_generation import generate_sidechain


RDLogger.DisableLog('rdApp.*')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_metrics(mol):
   
    if not mol: return None
    
    # A. 基础属性
    mw = Descriptors.MolWt(mol)       
    logp = Descriptors.MolLogP(mol)    
    hbd = Descriptors.NumHDonors(mol)  
    hba = Descriptors.NumHAcceptors(mol) 
    tpsa = Descriptors.TPSA(mol)       
    
    try:
        qed_score = QED.qed(mol)
    except:
        qed_score = 0.0
        
    # C. Lipinski Rule of 5 判断
    # 1. 分子量 <= 500
    # 2. LogP <= 5
    # 3. 氢键供体 <= 5
    # 4. 氢键受体 <= 10
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    is_lipinski_pass = (violations <= 1) 
    
    return {
        'MW': mw,
        'LogP': logp,
        'HBD': hbd,
        'HBA': hba,
        'TPSA': tpsa,
        'QED': qed_score,
        'Lipinski': is_lipinski_pass
    }

def clean_smiles(smi):
    if '.' in smi:
        fragments = smi.split('.')
        smi = max(fragments, key=len)
    return smi


def run_evaluation():
    print("🚀 加载模型...")
    DATA_PATH = '/home/myw/drugvae/3D_model/data/generation_data.pt'
    MODEL_PATH = '/home/myw/drugvae/3D_model/data/generator.pth'
    
    data_pkg = torch.load(DATA_PATH)
    vocab = data_pkg['vocab']
    char_to_idx = data_pkg['char_to_idx']
    idx_to_char = data_pkg['idx_to_char']
    
    model = ScaffoldGenerator(vocab_size=len(vocab), hidden_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

  
    test_data = data_pkg['data'][:50]
    
    results = []
    print(f"🧪 开始大规模海选 (筛选 {len(test_data)} 个骨架)...")
    
    for i, item in enumerate(test_data):
        core_pos = item['core_pos']
        core_z = item['core_z']

      
        best_qed_for_this_core = -1
        best_mol_info = None
        
        for attempt in range(5):
            raw_smi = generate_sidechain(model, core_pos, core_z, idx_to_char, char_to_idx, 
                                         max_len=80, temperature=0.7)
            clean_smi = clean_smiles(raw_smi)
            
            
            if not clean_smi.startswith('*'): display_smi = '*' + clean_smi
            else: display_smi = clean_smi
            
            mol = Chem.MolFromSmiles(display_smi)
            if mol:
                metrics = calculate_metrics(mol)
                if metrics:
                    
                    if metrics['QED'] > best_qed_for_this_core:
                        best_qed_for_this_core = metrics['QED']
                        metrics['SMILES'] = display_smi
                        metrics['Case_ID'] = i
                        best_mol_info = metrics
        
        if best_mol_info:
            results.append(best_mol_info)
            
            if (i+1) % 10 == 0:
                print(f"   ...已处理 {i+1} 个骨架")

  
    if not results:
        print("⚠️ 没有生成有效分子。")
        return

    df = pd.DataFrame(results)
     
    df_sorted = df.sort_values(by='QED', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("药物生成排行榜 (Top 10 By QED)")
    print("="*80)
    
     
    columns_to_show = ['Case_ID', 'SMILES', 'QED', 'LogP', 'MW', 'Lipinski']
    print(df_sorted[columns_to_show].head(10).to_string(index=False))
    
     
    df_sorted.to_csv('/home/myw/drugvae/3D_model/drug_candidates_ranked.csv', index=False)
    print(f"\n💾 完整榜单已保存至: 3D_model/drug_candidates_ranked.csv")
    
    
    avg_qed = df['QED'].mean()
    pass_rate = df['Lipinski'].mean() * 100
    print("-" * 50)
    print(f"📊 统计概览:")
    print(f"   平均 QED 得分: {avg_qed:.4f} (越接近1越好)")
    print(f"   五规则通过率: {pass_rate:.1f}%")
    print("-" * 50)

if __name__ == "__main__":
    run_evaluation()
