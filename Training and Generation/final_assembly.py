import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import os


RDLogger.EnableLog('rdApp.*')

# 引用模型
from generate_model import ScaffoldGenerator
from run_generation import generate_sidechain

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



TARGET_CORE_SMILES = "*c1ccc(N2CC(O*)OC2=O)cc1" 

NUM_SAMPLES = 20
OUTPUT_CSV = "/home/myw/drugvae/3D_model/data/multi_point_results.csv"

DATA_PATH = '/home/myw/drugvae/3D_model/data/generation_data.pt'
MODEL_PATH = '/home/myw/drugvae/3D_model/data/generator.pth'

GENERATION_TEMP = 0.5 
MAX_RETRIES = 50 
# ==============================================

def get_3d_input_robust(smi):
    try:
        temp_smi = smi.replace('*', 'C')
        mol = Chem.MolFromSmiles(temp_smi)
        if not mol: return None, None
        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res == -1:
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        
        mol_star = Chem.MolFromSmiles(smi)
        mol_star = Chem.AddHs(mol_star)
        AllChem.EmbedMolecule(mol_star, useRandomCoords=True)
        
        conf = mol_star.GetConformer()
        pos_list = []
        z_list = []
        for i in range(mol_star.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            pos_list.append([pos.x, pos.y, pos.z])
            z_list.append(mol_star.GetAtomWithIdx(i).GetAtomicNum())
            
        core_pos = torch.tensor(pos_list, dtype=torch.float32)
        core_z = torch.tensor(z_list, dtype=torch.long)
        core_pos = core_pos - core_pos.mean(dim=0, keepdim=True)
        return core_pos, core_z
    except:
        return None, None

def is_valid_smiles(smi):
    if not smi: return False
    try:
        check_smi = smi.replace('*', 'C')
        m = Chem.MolFromSmiles(check_smi)
        return True if m else False
    except:
        return False

def generate_valid_sidechain_with_retry(model, core_pos, core_z, idx_to_char, char_to_idx):
    for attempt in range(MAX_RETRIES):
        raw_smi = generate_sidechain(model, core_pos, core_z, idx_to_char, char_to_idx, temperature=GENERATION_TEMP)
        # 允许生成空侧链
        if raw_smi == '*' or raw_smi == '*.*':
            return raw_smi
            
        if '.' in raw_smi:
            frags = raw_smi.split('.')
            if all(is_valid_smiles(f) for f in frags):
                return raw_smi 
        else:
            if is_valid_smiles(raw_smi):
                return raw_smi 
    return None

def stitch_smart_v2(core_smi, side_smi_str):

    try:
        if '.' in side_smi_str:
            fragments = side_smi_str.split('.')
        else:
            fragments = [side_smi_str]
        
        
        clean_frags = [f.replace('*', '') for f in fragments]
        
        current_smi = core_smi
        
        
        while '*' in current_smi and clean_frags:
           
            idx = current_smi.find('*')
            
           
            frag = clean_frags.pop(0)
            
            if idx == 0:
                
                current_smi = frag + current_smi[1:]
                
            
            elif idx > 0 and idx + 1 < len(current_smi) and \
                 current_smi[idx-1] == '(' and current_smi[idx+1] == ')':
                current_smi = current_smi[:idx] + frag + current_smi[idx+1:]
                
            
            else:
                
                replacement = f"({frag})" if frag else "" 
                current_smi = current_smi[:idx] + replacement + current_smi[idx+1:]
        

        current_smi = current_smi.replace('*', '')
     
        current_smi = current_smi.replace('()', '')
        
     
        mol = Chem.MolFromSmiles(current_smi)
        if mol:
            try:
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol)
            except:
                return None
        return None

    except Exception as e:
        print(f"Logic Error: {e}")
        return None

def main():
    print(f"🎯 目标骨架: {TARGET_CORE_SMILES}")
    
    if not os.path.exists(DATA_PATH): return

    data_pkg = torch.load(DATA_PATH)
    vocab = data_pkg['vocab']
    char_to_idx = data_pkg['char_to_idx']
    idx_to_char = data_pkg['idx_to_char']
    
    model = ScaffoldGenerator(vocab_size=len(vocab), hidden_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    core_pos, core_z = get_3d_input_robust(TARGET_CORE_SMILES)
    if core_pos is None: return

    print("正在生成...")
    print("-" * 60)
    
    results = []
    success_cnt = 0
    
    for i in range(NUM_SAMPLES):
        side_smi = generate_valid_sidechain_with_retry(model, core_pos, core_z, idx_to_char, char_to_idx)
        
        full_smi = None
        status = "失败"
        
        if side_smi:
            # 使用 V2 版本的拼接
            full_smi = stitch_smart_v2(TARGET_CORE_SMILES, side_smi)
            
            if full_smi:
                try:
                    mol = Chem.MolFromSmiles(full_smi)
                    if mol:
                        qed = QED.qed(mol)
                        logp = Descriptors.MolLogP(mol)
                        status = "成功"
                        success_cnt += 1
                        results.append({
                            'Full_Molecule': full_smi, 
                            'QED': round(qed, 3),
                            'LogP': round(logp, 2)
                        })
                except:
                    pass
        
        disp = full_smi if full_smi else "---"
        print(f"样本 {i+1:02d} | {status} | {disp[:40]}...")

    if results:
        df = pd.DataFrame(results).sort_values(by='QED', ascending=False)
        df.to_csv(OUTPUT_CSV, index=False)
        print("-" * 60)
        print(f"🎉 成功率: {success_cnt}/{NUM_SAMPLES}")
        print(f"💾 结果已保存至 {OUTPUT_CSV}")
        print("🏆 Top 3:")
        print(df.head(3).to_string(index=False))
    else:
        print("⚠️ 依然失败。请检查是否还有其他 SMILES 语法问题。")

if __name__ == "__main__":
    main()
