import pandas as pd

def load_data(file_path):
    df = pd.read_excel(file_path)
    description = df[['ID','Job Description']]
    df = df.drop(columns=['Tags', 'Job Description'], axis=1)
    df['Job Location'] = df['Job Location'].fillna('Remote')
    df = df.drop_duplicates()
    return df, description
    
file_path = 'D:/Assignment/RAG/Data/raw/LF Jobs.xlsx'
df, description = load_data(file_path)
# save the cleaned dataframe to a new Excel file
output_file_path = 'D:/Assignment/RAG/Data/cleaned/'
df.to_excel(output_file_path + 'metadata.xlsx', index=False)
description.to_excel(output_file_path + 'description.xlsx', index=False)
print("Data loaded and cleaned successfully. Cleaned metadata saved to:", output_file_path)