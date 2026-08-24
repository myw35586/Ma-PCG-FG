import pandas as pd
import torch
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

INPUT_FILE = 'data/cleaned_antibiotics_for_training.csv'
OUTPUT_FILE = 'data/processed_antibiotics_3d.pt'


def generate_3d_data(input_path, output_path):

    if not os.path.exists(input_path):
        print(f"错误：找不到文件 {input_path}")
        return

    print(f"正在读取数据: {input_path} ...")
    df = pd.read_csv(input_path)
    
    data_list = []
    success_count = 0
    fail_count = 0
    
    print(f"开始处理 {len(df)} 个分子 (生成 3D 构象 + 拆分骨架)...")


    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            smiles = row['SMILES']
            core_smarts = row['core_smarts']

            label = float(row['XLogP']) 

            # A. 构建分子对象
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol) # 3D 必须加氢
            
            core = Chem.MolFromSmarts(core_smarts)
            
            if mol is None or core is None:
                fail_count += 1
                continue


            embed_res = AllChem.EmbedMolecule(mol, AllChem.ETKDG(randomSeed=42))
            if embed_res != 0:
                embed_res = AllChem.EmbedMolecule(mol, AllChem.ETKDG(useRandomCoords=True))
                if embed_res != 0:
                    fail_count += 1
                    continue
            

            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass


            matches = mol.GetSubstructMatches(core)
            if not matches:
                fail_count += 1
                continue
            

            core_indices = set(matches[0])
            all_indices = set(range(mol.GetNumAtoms()))
            sidechain_indices = all_indices - core_indices

            if len(sidechain_indices) == 0:

                fail_count += 1 
                continue


            conf = mol.GetConformer()
            

            core_pos = []
            core_atom_nums = []
            for idx in core_indices:
                pos = conf.GetAtomPosition(idx)
                core_pos.append([pos.x, pos.y, pos.z])
                core_atom_nums.append(mol.GetAtomWithIdx(idx).GetAtomicNum())
            

            sidechain_pos = []
            sidechain_atom_nums = []
            for idx in sidechain_indices:
                pos = conf.GetAtomPosition(idx)
                sidechain_pos.append([pos.x, pos.y, pos.z])
                sidechain_atom_nums.append(mol.GetAtomWithIdx(idx).GetAtomicNum())


            data_item = {
                'smiles': smiles,
                # 骨架特征
                'core_pos': torch.tensor(core_pos, dtype=torch.float32),
                'core_z': torch.tensor(core_atom_nums, dtype=torch.long), # 原子序数
                # 侧链特征
                'sidechain_pos': torch.tensor(sidechain_pos, dtype=torch.float32),
                'sidechain_z': torch.tensor(sidechain_atom_nums, dtype=torch.long), # 原子序数
                # 标签
                'y': torch.tensor([label], dtype=torch.float32)
            }
            
            data_list.append(data_item)
            success_count += 1

        except Exception as e:

            fail_count += 1
            continue


    print("-" * 30)
    print(f"处理完成！")
    print(f"成功: {success_count} 条")
    print(f"失败/跳过: {fail_count} 条")
    
    if success_count > 0:
        torch.save(data_list, output_path)
        print(f"数据已保存至: {output_path}")
        print("可以直接用于 PyG 模型训练了。")
    else:
        print("没有生成有效数据，请检查输入 CSV。")

if __name__ == "__main__":
    generate_3d_data(INPUT_FILE, OUTPUT_FILE)
