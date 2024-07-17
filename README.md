# Futures Trading Analytics Model

This repository contains an analytics model developed to analyze specific blocks of data, including price, volume, and time data, to determine if these blocks can predict a minimum level of price movement up or down. The data covers approximately 6 months and has been predefined and extracted for analysis.

## Project Description

As a futures trader, developing automated trading algorithms is essential for predicting market movements. This project aims to leverage machine learning techniques to analyze trading data and identify patterns that could indicate potential price movements.

## Data Description

The data used in this project includes:
- **Price Data**: The price of the asset at different time intervals.
- **Volume Data**: The volume of buy and sell orders at each price level.
- **Time Data**: Timestamps for each data point.

The data is organized into blocks, each representing a specific period. Each block is further divided into bars, representing smaller time intervals within the block.

## Analysis Steps

1. **Data Loading and Initial Processing**:
    - Load the dataset from a CSV file.
    - Display the first few rows of the dataset for an initial overview.
    - Display the data information to understand the structure and types of data.

2. **Volume Calculation**:
    - Calculate the total buy and sell volumes for each bar.
    - Aggregate these volumes to calculate the total volume for each block.

3. **Volume Profile Creation**:
    - Create volume profiles for each block, showing the distribution of buy and sell volumes at different price levels.
    - Save these volume profiles to CSV files for further analysis.

4. **Identifying Key Volume Points**:
    - Identify the price level with the highest total volume in each block.
    - Calculate the buy and sell volumes at this price level.
    - Determine the volume difference (buy volume - sell volume) at this price level.

5. **Close Direction Analysis**:
    - Determine the direction of price movement (up, down, or equal) for the bar with the highest total volume in each block.
    - Repeat the analysis for the bar with the second highest total volume, if available.

6. **Divergence Analysis**:
    - Analyze the divergence between the volume direction and the price movement direction.
    - Identify blocks where the highest volume bar's buy volume is greater than the sell volume, but the price moves down (and vice versa).

7. **Breakout Volume Analysis**:
    - Calculate the total volume for breakout bars (bars with a specific identifier).

8. **Export Results**:
    - Export the analyzed data to CSV files for further use in trading algorithms.

## Implementation

The following Python code performs the analysis described above. The code is organized into two sets: one excluding bars with a specific identifier (`-1`) and one including them.

```python
# Set One (-1 bars excluded)
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
df =  pd.read_csv('bar_data_03-2023_thru_07-2023.csv')
df.head(10)

# Data information
df.info()

# Volume calculations
df['bar_buy_vol'] = df['buy0']
df['bar_sell_vol'] = df['sell0']
for idx in range(1,7):
    df['bar_buy_vol'] += df[f'buy{idx}']
    df['bar_sell_vol'] += df[f'sell{idx}']
df['bar_total_vol'] = df['bar_buy_vol'] + df['bar_sell_vol']

# Aggregate volume for each block
for i in df.index:
    block_df = df[df.block_id == df.loc[i,'block_id']]
    grouped_block_buy_vol = block_df[block_df.bar_number != -1]['bar_buy_vol'].sum()
    grouped_block_sell_vol = block_df[block_df.bar_number != -1]['bar_sell_vol'].sum()
    total_block_vol = grouped_block_buy_vol + grouped_block_sell_vol
    df.loc[i,'bock_buy_vol'] = grouped_block_buy_vol
    df.loc[i,'bock_sell_vol'] = grouped_block_sell_vol
    df.loc[i,'bock_total_vol'] = total_block_vol

# Create volume profiles for each block
block_vol_profiles = {}
for block in df.block_id.unique():
    block_vol_profile = pd.DataFrame()
    block_df = df[df.block_id == block]
    block_df = block_df[block_df.bar_number != -1]
    bar_prices = {}
    for bar_number in block_df.bar_number:
        prices = block_df[block_df.bar_number == bar_number][[col for col in block_df if 'price' in col]].T.iloc[:, 0].to_list()
        buys = block_df[block_df.bar_number == bar_number][[col for col in block_df if 'buy' in col and '_' not in col]].T.iloc[:, 0].to_list()
        sells = block_df[block_df.bar_number == bar_number][[col for col in block_df if 'sell' in col and '_' not in col]].T.iloc[:, 0].to_list()
        for price, buy, sell in zip(prices, buys, sells):
            try:
                block_vol_profile.loc[price, 'buy'] += buy
            except:
                block_vol_profile.loc[price, 'buy'] = buy
            block_vol_profile.fillna(0, inplace=True)
            try:
                block_vol_profile.loc[price, 'sell'] += sell
            except:
                block_vol_profile.loc[price, 'sell'] = sell
            try:
                block_vol_profile.loc[price, 'total_vol'] += buy + sell
            except:
                block_vol_profile.loc[price, 'total_vol'] = buy + sell
    block_vol_profile.fillna(0, inplace=True)
    block_vol_profile.reset_index(inplace=True)
    block_vol_profile.rename(columns={'index': 'prices'}, inplace=True)
    block_vol_profiles[block] = block_vol_profile

# Save volume profiles to CSV
import os
for block, profile in block_vol_profiles.items():
    profile['block_id'] = block
    profile.to_csv(os.path.join('profiles', f'{block.replace(":", "-")}.csv'), index=False)

# Update original DataFrame with volume profile data
for idx in range(len(df)):
    block_id = df.loc[idx, 'block_id']
    block_profile = block_vol_profiles[block_id]
    max_price = block_profile.total_vol.max()
    df.loc[idx, 'buy_vol_at_high_price'] = block_profile[block_profile.total_vol == max_price].buy.iloc[0]
    df.loc[idx, 'sell_vol_at_high_price'] = block_profile[block_profile.total_vol == max_price].sell.iloc[0]
    df.loc[idx, 'vol_diff_at_high_price'] = df.loc[idx, 'buy_vol_at_high_price'] - df.loc[idx, 'sell_vol_at_high_price']

# Close direction analysis
close_directions = {}
for block in df.block_id.unique():
    block_df = df[df.block_id == block]
    block_df = block_df[block_df.bar_number != -1]
    bar_number = block_df.sort_values(by='bar_total_vol', ascending=False).reset_index().loc[0, 'bar_number']
    close_direction = block_df.sort_values(by='bar_total_vol', ascending=False).reset_index().loc[0, 'close'] - block_df.sort_values(by='bar_total_vol', ascending=False).reset_index().loc[0, 'open']
    if close_direction > 0:
        close_directions[block] = ['Up', bar_number]
    elif close_direction < 0:
        close_directions[block] = ['Down', bar_number]
    else:
        close_directions[block] = ['Equal', bar_number]

close_directions_2 = {}
for block in df.block_id.unique():
    block_df = df[df.block_id == block]
    block_df = block_df[block_df.bar_number != -1]
    if len(block_df.sort_values(by='bar_total_vol', ascending=False).reset_index()) > 1:
        bar_number = block_df.sort_values(by='bar_total_vol', ascending=False).reset_index().loc[1, 'bar_number']
        close_direction = block_df.sort_values(by='bar_total_vol', ascending=False).reset_index().loc[1, 'close'] - block_df.sort_values(by='bar_total_vol', ascending=False).reset_index().loc[1, 'open']
        if close_direction > 0:
            close_directions_2[block] = ['Up', bar_number]
        elif close_direction < 0:
            close_directions_2[block] = ['Down', bar_number]
        else:
            close_directions_2[block] = ['Equal', bar_number]

# Update DataFrame with close direction data
for i in range(len(df)):
    block = df.loc[i, 'block_id']
    df.loc[i, 'close_direction_highest'] = close_directions[block][0]
    df.loc[i, 'bar_with_greatest_total_vol'] = close_directions[block][1]
    try:
        df.loc[i, 'close_direction_2nd_highest'] = close_directions_2[block][0]
        df.loc[i, 'bar_with_2nd_greatest_total_vol'] = close_directions_2[block][1]
    except:
        pass

# Divergence analysis
for i in range(len(df)):
    block = df.loc[i, 'block_id']
    if df.loc[i, 'close_direction_highest'] == 'Equal':
        df.loc[i, 'divergence_of_highest_vol'] = None
    elif (

df.loc[i, 'vol_diff_at_high_price'] > 0) & (df.loc[i, 'close_direction_highest'] == 'Down'):
        df.loc[i, 'divergence_of_highest_vol'] = 'Positive_Volume_Negative_Close'
    elif (df.loc[i, 'vol_diff_at_high_price'] < 0) & (df.loc[i, 'close_direction_highest'] == 'Up'):
        df.loc[i, 'divergence_of_highest_vol'] = 'Negative_Volume_Positive_Close'
    else:
        df.loc[i, 'divergence_of_highest_vol'] = None

# Calculate breakout volume
breakout_vols = {}
for block in df.block_id.unique():
    block_df = df[df.block_id == block]
    breakout_vol = block_df[block_df.bar_number == -1].bar_total_vol.sum()
    breakout_vols[block] = breakout_vol

for i in range(len(df)):
    block = df.loc[i, 'block_id']
    df.loc[i, 'breakout_vol'] = breakout_vols[block]

# Export analyzed data
df.to_csv('analyzed_data.csv', index=False)
```

## Set Two (Including -1 Bars)

```python
# Re-run the analysis steps with -1 bars included as shown in the code above
# Use the modified code as per the second set requirements

# Load and process the dataset similarly as shown above
# Ensure the analysis includes bars with bar_number == -1 where required
# Save the output for Set Two as 'analyzed_data_set_two.csv'
```

## How to Use

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/futures-trading-analytics-model.git
    cd futures-trading-analytics-model
    ```

2. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the analysis**:
    ```sh
    python analysis_script.py
    ```

4. **Review the output**:
    - `analyzed_data.csv`: Results of Set One (excluding -1 bars).
    - `analyzed_data_set_two.csv`: Results of Set Two (including -1 bars).

## Future Work

- Implement additional features for advanced trading strategy development.
- Improve the model by incorporating more data points and refining the analysis techniques.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License.
