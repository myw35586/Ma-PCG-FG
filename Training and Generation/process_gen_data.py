import pandas as pd
import torch
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


CSV_FILE = 'data/cleaned_antibiotics_for_training.csv' 
try:
    pd.read_csv(CSV_FILE)
except FileNotFoundError:
    CSV_FILE = 'data/cleaned_antibiotics_for_training.csv'

OUTPUT_FILE = 'data/generation_data.pt'


df = pd.read_csv(CSV_FILE)
print(f"📄 读取到 {len(df)} 条原始数据")

data_list = []
all_chars = set()

print("⚗️ 开始处理数据 (生成3D骨架 + 提取侧链序列)...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    full_smi = row['SMILES']
    core_smart = row['core_smarts']
    

    mol = Chem.MolFromSmiles(full_smi)
    core_query = Chem.MolFromSmarts(core_smart)
    
    if not mol or not core_query or not mol.HasSubstructMatch(core_query):
        continue
        
    try:

        sidechains = Chem.ReplaceCore(mol, core_query)
        if not sidechains: continue
        

        target_smi = Chem.MolToSmiles(sidechains, isomericSmiles=False)
 
        target_smi = re.sub(r'\[\d+\*\]', '*', target_smi)
       
        all_chars.update(list(target_smi))
    except:
        continue

    
    core_mol = Chem.MolFromSmiles(core_smart)
    
    if not core_mol:
        core_mol = Chem.MolFromSmarts(core_smart)
        if core_mol:
            
            core_mol.UpdatePropertyCache(strict=False)
    
    if not core_mol: continue
    
    
    try:
        core_mol = Chem.AddHs(core_mol) 
        
       
        res = AllChem.EmbedMolecule(core_mol, AllChem.ETKDG())
        if res == -1: 
            params = AllChem.ETKDG()
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(core_mol, params)
            
        if res == -1: continue 
        
        AllChem.MMFFOptimizeMolecule(core_mol)
    except Exception as e:
        # print(f"3D Error: {e}")
        continue 
        
    conf = core_mol.GetConformer()
    pos_list = []
    z_list = []
    for i in range(core_mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        pos_list.append([pos.x, pos.y, pos.z])
        z_list.append(core_mol.GetAtomWithIdx(i).GetAtomicNum())
        

    core_pos = torch.tensor(pos_list, dtype=torch.float32)
    core_z = torch.tensor(z_list, dtype=torch.long)
    

    if core_pos.shape[0] > 0:
        core_pos = core_pos - core_pos.mean(dim=0, keepdim=True)
    else:
        continue


    data_list.append({
        'core_pos': core_pos,
        'core_z': core_z,
        'target_smi': target_smi
    })


special_tokens = ['<PAD>', '<SOS>', '<EOS>']
chars = sorted(list(all_chars))
vocab = special_tokens + chars


char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for i, c in enumerate(vocab)}

print(f"\n✅ 处理完成！有效数据: {len(data_list)} 条")
print(f"📚 词表大小: {len(vocab)}")
print(f"💾 保存至 {OUTPUT_FILE} ...")

torch.save({
    'data': data_list,
    'vocab': vocab,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char
}, OUTPUT_FILE)
