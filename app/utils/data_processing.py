import pandas as pd

class DataProcessor:
    def load_and_clean_data(self, file):
        df = pd.read_excel(file)
        df.columns = ['Date', 'Level']
        df['Date'] = pd.to_datetime(df['Date'])
        df['Level'] = pd.to_numeric(df['Level'])
        return df
