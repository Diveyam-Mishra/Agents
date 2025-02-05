import pandas as pd
import os
from collections import defaultdict

root_dir = "/kaggle/input/supply-mint"

file_mapping = defaultdict(dict)

for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        parts = file.split('_')
        if len(parts) >= 3:
            prefix = parts[0]
            category = parts[1]
            file_mapping[prefix][category] = os.path.join(dirpath, file)

df_item=pd.read_csv('/kaggle/input/supply-mint/jk_item_202501211230.csv')
df_stock=pd.read_csv('/kaggle/input/supply-mint/jk_stock_202501211230.csv')
df_sales=pd.read_csv('/kaggle/input/supply-mint/jk_sales_202501211230.csv')
df_item.dropna(axis=1, how='all', inplace=True)
df_stock.dropna(axis=1, how='all', inplace=True)
df_sales.dropna(axis=1, how='all', inplace=True)
columns_to_remove = ['icode','cost_amt','qty','desc1','desc2','desc3','desc4','desc5','desc6','udfnum01',
    'udfnum02','udfnum03','udfnum04','udfnum05','last_changed','stockindate','slcode','slname','hsncode',
    'invhsnsacmain_code','ext','ws_image1_url','sm_itemcode','sm_integer_itemcode',
    'ccode2', 'cname2', 'ccode3', 'cname3', 'ccode5', 'ccode6','id', 'cost_amt','shrtname',
    'hl1code', 'hl1name', 'hl2code', 'hl2name', 'hl4code', 'article_mrp','hl3code','ccode4','cost_rate',
    'hl4name', 'hl5code', 'hl6code', 'itemname', 'image_url', 'wsp',
    'barcode', 'taxcode', 'uom', 'rate', 'article_mrprangefrom', 'cost_rate','cname4','cname5','cname6',
    'article_mrprangeto', 'price_basis', 'basis', 'ccode1', 'cname1','is_promo','discount_amt','tax_amt',]

columns_to_remove_in_item = [col for col in columns_to_remove if col in df_item.columns]
df_item = df_item.drop(columns=columns_to_remove)
columns_to_remove_in_sales = [col for col in columns_to_remove if col in df_sales.columns]
df_item['icode']=df_item['itemcode']
df= pd.merge(df_sales,df_item, on='icode')
df['date'] = pd.to_datetime(df['date'])


def map_shade(color):
    color = str(color).strip().lower()
    if color in ['beige', 'brown', 'cream', 'chikoo', 'offwhite', 'grey', 'gray', 'ash', 'fawn', 'stone']:
        return 'Neutral'
    elif color in ['white', 'peach', 'lavender', 'mauve', 'baby pink', 'pale mauve', 'light pink', 'mint green', 
                   'lilac', 'pastel yellow', 'powder blue', 'aqua', 'light green']:
        return 'Pastel'
    elif color in ['black', 'navy', 'navy blue', 'maroon', 'burgundy', 'rust', 'dark blue', 'charcoal', 'cadbury', 
                   'wine', 'dark brown', 'dark grey', 'dark olive', 'dark khaki']:
        return 'Dark'
    elif color in ['red', 'blue', 'orange', 'yellow', 'green', 'purple', 'violet', 'fuchsia', 'magenta', 
                   'lime', 'gold', 'silver', 'coral', 'mustard']:
        return 'Bright'
    elif 'multi' in color or 'pattern' in color or 'stripe' in color or 'check' in color or 'print' in color:
        return 'Patterned'
    elif color in ['gold', 'silver', 'bronze', 'copper', 'metallic']:
        return 'Metallic'
    else:
        return 'Other'
df['mapped_shade'] = df['cname5'].apply(map_shade)

