import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import os

# 引用模型
from generate_model import ScaffoldGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_sidechain(model, core_pos, core_z, idx_to_char, char_to_idx, max_len=50, temperature=1.0):
    
    model.eval()
    with torch.no_grad():
        
        core_batch = torch.zeros(core_z.size(0), dtype=torch.long).to(DEVICE)
        core_pos = core_pos.to(DEVICE)
        core_z = core_z.to(DEVICE)
        
        context = model.encode_core(core_pos, core_z, core_batch)
        
        current_token = torch.tensor([[char_to_idx['<SOS>']]], dtype=torch.long).to(DEVICE)

        hidden = context.unsqueeze(0) # [1, 1, Hidden]
        
        generated_indices = []
        
        for _ in range(max_len):
 
            embed = model.embedding(current_token)
            
            output, hidden = model.rnn(embed, hidden)
            
            logits = model.fc_out(output) # [1, 1, Vocab]
            logits = logits.squeeze(0).squeeze(0)
            
            if temperature == 0:
                next_token_idx = torch.argmax(logits).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token_idx = torch.multinomial(probs, 1).item()
            
            if idx_to_char[next_token_idx] == '<EOS>':
                break
                
            generated_indices.append(next_token_idx)
            
            current_token = torch.tensor([[next_token_idx]], dtype=torch.long).to(DEVICE)
            
    gen_str = "".join([idx_to_char[idx] for idx in generated_indices])
    return gen_str

# ================= 2. 拼接与验证工具 =================
def attach_generated_sidechain(core_smi, side_smi):

    try:
        if '*' not in side_smi:
            
            side_smi = '*' + side_smi
            
        core_mol = Chem.MolFromSmiles(core_smi)
        side_mol = Chem.MolFromSmiles(side_smi)
        
        if not core_mol or not side_mol: return None
        

        rxn = AllChem.ReactionFromSmarts('[*:1]-[#0].[*:2]-[#0]>>[*:1]-[*:2]')
        ps = rxn.RunReactants((core_mol, side_mol))
        
        if ps:
            product = ps[0][0]
            Chem.SanitizeMol(product)
            return Chem.MolToSmiles(product)
        return None
    except:
        return None

# ================= 3. 主程序 =================
def run():
    print("🚀 加载模型和词表...")
    DATA_PATH = '/home/myw/drugvae/3D_model/data/generation_data.pt' # 为了读词表
    MODEL_PATH = '/home/myw/drugvae/3D_model/data/generator.pth'
    
    # 加载词表
    data_pkg = torch.load(DATA_PATH)
    vocab = data_pkg['vocab']
    char_to_idx = data_pkg['char_to_idx']
    idx_to_char = data_pkg['idx_to_char']
    
    # 加载模型
    model = ScaffoldGenerator(vocab_size=len(vocab), hidden_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    print("🧪 准备骨架...")
    test_data = data_pkg['data'][:5] 
    
    print("-" * 60)
    print(f"{'Core Skeleton':<30} | {'AI Generated Sidechain':<25}")
    print("-" * 60)
    
    for i, item in enumerate(test_data):
        core_pos = item['core_pos']
        core_z = item['core_z']
        
        print(f"🧬 Case {i+1}:")
        
        for t in range(3):
            gen_smi = generate_sidechain(model, core_pos, core_z, idx_to_char, char_to_idx, temperature=0.8)
            print(f"   Attempt {t+1}: {gen_smi}")
            
    print("-" * 60)
    print("✅ 生成演示完毕！")

if __name__ == "__main__":
    run()
