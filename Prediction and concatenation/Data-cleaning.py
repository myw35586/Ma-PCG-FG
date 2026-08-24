import pandas as pd
import os

# 1. 设置文件路径
input_file_path = 'data/antibiotics_final_results.csv'
output_file_path = 'data/cleaned_antibiotics_for_training.csv'


if not os.path.exists(input_file_path):
    print(f"错误：找不到文件 {input_file_path}，请确认路径是否正确。")
else:
    # 读取 CSV
    df = pd.read_csv(input_file_path)

    print(f"清洗前总数: {len(df)}")
    print("-" * 30)


    clean_df = df[df['core_type'] != 'unmatched'].copy()


    class_counts = clean_df['core_type'].value_counts()
    valid_classes = class_counts[class_counts > 10].index
    clean_df = clean_df[clean_df['core_type'].isin(valid_classes)]


    print(f"清洗后总数: {len(clean_df)}")
    print("-" * 30)
    print("清洗后各类别分布 (Core Type Distribution):")
    print(clean_df['core_type'].value_counts())

    clean_df.to_csv(output_file_path, index=False)
    print("-" * 30)
    print(f"处理完成！清洗后的文件已保存至: {output_file_path}")
