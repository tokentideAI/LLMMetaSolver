import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import json


def solve_mad_optimization():
    # 1. Read instance.json (optional, mainly for parameter verification)
    with open('instance.json', 'r', encoding='utf-8') as f:
        instance_info = json.load(f)

    print(f"Solving problem: {instance_info.get('id', 'MAD Problem')}")

    # 2. Read CSV data
    # Note: According to generation logic, CSV files usually do not have headers, we need to manually specify column names

    # C1: Basic Stock Info (Stock_ID, Industry_ID, Sector_ID)
    df_c1 = pd.read_csv('data/C1.csv')

    # C2: Sector Limits (Sector_ID, Limit)
    df_c2 = pd.read_csv('data/C2.csv').rename(columns={'MaxCount': 'Limit'})

    # C3 is a wide matrix: rows are metrics and columns are Stock_0, Stock_1, ...
    c3_wide = pd.read_csv('data/C3.csv', index_col=0)
    c3_wide.index.name = 'MetricID'
    df_c3 = c3_wide.reset_index().melt(
        id_vars='MetricID', var_name='StockID', value_name='Score'
    )
    df_c3['StockID'] = df_c3['StockID'].str.extract(r'(\d+)$')[0].astype(int)

    # C4 is a wide matrix: rows are indicators and columns are Stock_0, Stock_1, ...
    c4_wide = pd.read_csv('data/C4.csv', index_col=0)
    c4_wide.index.name = 'IndicatorID'
    df_c4 = c4_wide.reset_index().melt(
        id_vars='IndicatorID', var_name='StockID', value_name='Sensitivity'
    )
    df_c4['IndicatorID'] = df_c4['IndicatorID'].str.extract(r'(\d+)$')[0].astype(int)
    df_c4['StockID'] = df_c4['StockID'].str.extract(r'(\d+)$')[0].astype(int)

    # C5: Economic Indicator Target Values (Indicator_ID, Target)
    df_c5 = pd.read_csv('data/C5.csv').rename(columns={'TargetValue': 'Target'})

    # 3. Initialize Gurobi model
    model = gp.Model("IndexTracking_MAD")

    # Get sets
    stocks = df_c1['StockID'].unique()
    indicators = df_c5['IndicatorID'].unique()

    # 4. Define decision variables
    # x[i]: Binary variable, whether to select stock i
    x = model.addVars(stocks, vtype=GRB.BINARY, name="x")

    # y_pos[j], y_neg[j]: Continuous variables, positive and negative deviation for indicator j
    # Used to linearize absolute value objective function |Expr - Target|
    y_pos = model.addVars(indicators, vtype=GRB.CONTINUOUS, lb=0.0, name="y_pos")
    y_neg = model.addVars(indicators, vtype=GRB.CONTINUOUS, lb=0.0, name="y_neg")

    # 5. Add constraints

    # (1) Industry constraints: Exactly one stock selected per industry
    # Group by IndustryID
    industry_groups = df_c1.groupby('IndustryID')
    for ind_id, group in industry_groups:
        model.addConstr(
            gp.quicksum(x[sid] for sid in group['StockID']) == 1,
            name=f"Industry_Select_{ind_id}"
        )

    # (2) Sector constraints: Number of selected stocks in each sector <= limit
    # Convert limits to dictionary for easy lookup
    sector_limit_dict = dict(zip(df_c2['SectorID'], df_c2['Limit']))
    sector_groups = df_c1.groupby('SectorID')
    for sec_id, group in sector_groups:
        if sec_id in sector_limit_dict:
            limit = sector_limit_dict[sec_id]
            model.addConstr(
                gp.quicksum(x[sid] for sid in group['StockID']) <= limit,
                name=f"Sector_Limit_{sec_id}"
            )

    # (3) Quality constraints: Weighted total score >= 0
    # Group by MetricID
    metric_groups = df_c3.groupby('MetricID')
    for met_id, group in metric_groups:
        # Only sum for stocks in the stock pool
        expr = gp.quicksum(row['Score'] * x[row['StockID']]
                           for _, row in group.iterrows() if row['StockID'] in stocks)
        model.addConstr(expr >= 0, name=f"Quality_Threshold_{met_id}")

    # (4) MAD constraints: Calculate deviation from target
    # Formula: Sum(Sensitivity * x) - Target = y_pos - y_neg
    # This is equivalent to: Sum(Sensitivity * x) + y_neg - y_pos = Target
    target_dict = dict(zip(df_c5['IndicatorID'], df_c5['Target']))

    # To build constraints, we need to group C4 by IndicatorID
    indicator_groups = df_c4.groupby('IndicatorID')

    for ind_id in indicators:
        target_val = target_dict.get(ind_id, 0.0)

        # Get sensitivities of all stocks for this indicator
        if ind_id in indicator_groups.groups:
            group = indicator_groups.get_group(ind_id)
            sens_expr = gp.quicksum(row['Sensitivity'] * x[row['StockID']]
                                    for _, row in group.iterrows() if row['StockID'] in stocks)
        else:
            sens_expr = 0

        # Add deviation constraint
        model.addConstr(
            sens_expr + y_neg[ind_id] - y_pos[ind_id] == target_val,
            name=f"Deviation_Calc_{ind_id}"
        )

    # 6. Set objective function
    # Minimize total absolute deviation (Mean Absolute Deviation, here actually Sum AD, optimization result is equivalent)
    model.setObjective(
        gp.quicksum(y_pos[j] + y_neg[j] for j in indicators),
        GRB.MINIMIZE
    )

    # 7. Solve and output results
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("\n" + "=" * 30)
        print(f"Optimal Objective Value (Total Absolute Deviation): {model.ObjVal}")
        print("=" * 30)

        # Print selected stocks
        selected_stocks = [i for i in stocks if x[i].X > 0.5]
        print(f"Number of selected stocks: {len(selected_stocks)}")
        print(f"Selected Stock IDs: {selected_stocks}")

    else:
        print("Optimal solution not found.")


if __name__ == "__main__":
    try:
        solve_mad_optimization()
    except Exception as e:
        print(f"Execution error: {e}")
        # If libraries are missing, prompt installation
        print("Please ensure gurobipy and pandas are installed: pip install gurobipy pandas")