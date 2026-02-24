#!/usr/bin/env python
"""
Interactive visualization of diseases in tycho_US dataset.
Produces interactive HTML files using Plotly for exploration.
"""

import torch
import numpy as np
import os
from collections import defaultdict
from datetime import datetime, timedelta

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    print("Plotly not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'plotly'])
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px


def load_tycho_data(data_path='tycho_US.pt'):
    """Load the tycho_US dataset."""
    if not os.path.isabs(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)
    
    print(f"Loading data from: {data_path}")
    data = torch.load(data_path, weights_only=False)
    return data


def aggregate_disease_data(disease_data):
    """
    Aggregate disease data across all states to get national time series.
    
    Returns:
        dict: {week_number: total_infections}
        dict: {state: {week_number: infections}}
    """
    national_data = defaultdict(float)
    state_data = {}
    
    for state, values in disease_data.items():
        state_data[state] = {}
        infections = values[0][0].numpy()
        time_weeks = values[0][1].numpy()
        
        for week, inf in zip(time_weeks, infections):
            week = int(week)
            national_data[week] += inf
            state_data[state][week] = inf
    
    return dict(national_data), state_data


def week_to_date(date_code):
    """Convert date code (YYYYMMDD format) to datetime object."""
    date_code = int(date_code)
    year = date_code // 10000
    month = (date_code % 10000) // 100
    day = date_code % 100
    # Handle edge cases
    if month < 1:
        month = 1
    if month > 12:
        month = 12
    if day < 1:
        day = 1
    if day > 28:  # Safe upper bound for all months
        day = 28
    try:
        return datetime(year, month, day)
    except ValueError:
        # Fallback for invalid dates
        return datetime(year, 1, 1)


def create_disease_overview_html(data, output_dir='visualizations'):
    """Create an overview HTML with all diseases."""
    os.makedirs(output_dir, exist_ok=True)
    
    diseases = sorted(data.keys())
    
    # Collect summary statistics
    summaries = []
    for disease in diseases:
        national, state_data = aggregate_disease_data(data[disease])
        weeks = sorted(national.keys())
        if len(weeks) == 0:
            continue
            
        total_cases = sum(national.values())
        max_cases = max(national.values())
        n_states = len(state_data)
        start_week = min(weeks)
        end_week = max(weeks)
        duration_years = (end_week - start_week) / 52
        
        summaries.append({
            'disease': disease,
            'total_cases': total_cases,
            'max_weekly_cases': max_cases,
            'n_states': n_states,
            'start_date': week_to_date(start_week).strftime('%Y-%m'),
            'end_date': week_to_date(end_week).strftime('%Y-%m'),
            'duration_years': duration_years,
            'n_weeks': len(weeks)
        })
    
    # Create overview figure with subplots for each disease
    n_diseases = len(summaries)
    n_cols = 3
    n_rows = (n_diseases + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[s['disease'] for s in summaries],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    colors = px.colors.qualitative.Set3
    
    for idx, summary in enumerate(summaries):
        disease = summary['disease']
        national, _ = aggregate_disease_data(data[disease])
        weeks = sorted(national.keys())
        values = [national[w] for w in weeks]
        dates = [week_to_date(w) for w in weeks]
        
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name=disease,
                line=dict(color=colors[idx % len(colors)], width=1),
                hovertemplate=f"<b>{disease}</b><br>" +
                              "Date: %{x}<br>" +
                              "Cases: %{y:,.0f}<extra></extra>"
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=dict(
            text="<b>Tycho US Disease Dataset Overview</b><br>" +
                 "<sup>National weekly case counts aggregated across all states</sup>",
            x=0.5,
            font=dict(size=20)
        ),
        height=300 * n_rows,
        showlegend=False,
        template='plotly_white'
    )
    
    # Save overview
    overview_path = os.path.join(output_dir, 'disease_overview.html')
    fig.write_html(overview_path, include_plotlyjs='cdn')
    print(f"Saved overview to: {overview_path}")
    
    # Create summary table
    create_summary_table_html(summaries, output_dir)
    
    return summaries


def create_summary_table_html(summaries, output_dir):
    """Create an HTML table with disease summaries."""
    
    # Sort by total cases
    summaries_sorted = sorted(summaries, key=lambda x: -x['total_cases'])
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Disease</b>', '<b>Total Cases</b>', '<b>Max Weekly</b>', 
                    '<b>States</b>', '<b>Start</b>', '<b>End</b>', '<b>Years</b>', '<b>Weeks</b>'],
            fill_color='#2c3e50',
            font=dict(color='white', size=12),
            align='left',
            height=35
        ),
        cells=dict(
            values=[
                [s['disease'] for s in summaries_sorted],
                [f"{s['total_cases']:,.0f}" for s in summaries_sorted],
                [f"{s['max_weekly_cases']:,.0f}" for s in summaries_sorted],
                [s['n_states'] for s in summaries_sorted],
                [s['start_date'] for s in summaries_sorted],
                [s['end_date'] for s in summaries_sorted],
                [f"{s['duration_years']:.1f}" for s in summaries_sorted],
                [s['n_weeks'] for s in summaries_sorted]
            ],
            fill_color=[['#f8f9fa', '#ffffff'] * (len(summaries_sorted) // 2 + 1)],
            font=dict(size=11),
            align='left',
            height=28
        )
    )])
    
    fig.update_layout(
        title=dict(
            text="<b>Disease Summary Statistics</b>",
            x=0.5,
            font=dict(size=18)
        ),
        height=50 + 35 * len(summaries_sorted)
    )
    
    table_path = os.path.join(output_dir, 'disease_summary_table.html')
    fig.write_html(table_path, include_plotlyjs='cdn')
    print(f"Saved summary table to: {table_path}")


def create_single_disease_html(data, disease, output_dir='visualizations'):
    """Create detailed interactive visualization for a single disease."""
    os.makedirs(output_dir, exist_ok=True)
    
    if disease not in data:
        print(f"Disease '{disease}' not found in dataset")
        return
    
    national, state_data = aggregate_disease_data(data[disease])
    weeks = sorted(national.keys())
    values = [national[w] for w in weeks]
    dates = [week_to_date(w) for w in weeks]
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"colspan": 2}, None]
        ],
        subplot_titles=[
            f'{disease} - National Weekly Cases',
            'State-Level Heatmap', 'Top 10 States by Total Cases',
            'Seasonal Pattern (Average by Month)'
        ],
        row_heights=[0.4, 0.35, 0.25],
        vertical_spacing=0.1
    )
    
    # 1. Main time series
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Weekly Cases',
            line=dict(color='#e74c3c', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)',
            hovertemplate="Date: %{x}<br>Cases: %{y:,.0f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add rolling average
    window = 12  # 12-week rolling average
    if len(values) > window:
        rolling_avg = np.convolve(values, np.ones(window)/window, mode='valid')
        rolling_dates = dates[window-1:]
        fig.add_trace(
            go.Scatter(
                x=rolling_dates,
                y=rolling_avg,
                mode='lines',
                name='12-week Moving Avg',
                line=dict(color='#2c3e50', width=2),
                hovertemplate="Date: %{x}<br>12-week Avg: %{y:,.0f}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # 2. State heatmap
    states = sorted(state_data.keys())
    all_weeks = sorted(set(w for sd in state_data.values() for w in sd.keys()))
    
    # Sample weeks for heatmap (every 4 weeks to reduce size)
    week_step = max(1, len(all_weeks) // 100)
    sampled_weeks = all_weeks[::week_step]
    
    heatmap_data = []
    for state in states:
        row_data = []
        for week in sampled_weeks:
            val = state_data[state].get(week, 0)
            row_data.append(val)
        heatmap_data.append(row_data)
    
    heatmap_dates = [week_to_date(w).strftime('%Y-%m') for w in sampled_weeks]
    
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=heatmap_dates,
            y=states,
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title='Cases', x=0.45, len=0.3, y=0.5),
            hovertemplate="State: %{y}<br>Date: %{x}<br>Cases: %{z:,.0f}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # 3. Top states bar chart
    state_totals = {state: sum(sd.values()) for state, sd in state_data.items()}
    top_states = sorted(state_totals.items(), key=lambda x: -x[1])[:10]
    
    fig.add_trace(
        go.Bar(
            x=[s[1] for s in top_states],
            y=[s[0] for s in top_states],
            orientation='h',
            marker=dict(color='#3498db'),
            hovertemplate="State: %{y}<br>Total Cases: %{x:,.0f}<extra></extra>"
        ),
        row=2, col=2
    )
    
    # 4. Seasonal pattern
    monthly_avg = defaultdict(list)
    for week, val in national.items():
        date = week_to_date(week)
        monthly_avg[date.month].append(val)
    
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    avg_by_month = [np.mean(monthly_avg[m]) if m in monthly_avg else 0 for m in months]
    
    fig.add_trace(
        go.Bar(
            x=month_names,
            y=avg_by_month,
            marker=dict(
                color=avg_by_month,
                colorscale='RdYlBu_r',
                showscale=False
            ),
            hovertemplate="Month: %{x}<br>Avg Cases: %{y:,.0f}<extra></extra>"
        ),
        row=3, col=1
    )
    
    # Update layout
    start_date = min(dates).strftime('%Y')
    end_date = max(dates).strftime('%Y')
    total_cases = sum(values)
    
    fig.update_layout(
        title=dict(
            text=f"<b>{disease} Analysis</b><br>" +
                 f"<sup>Period: {start_date}-{end_date} | Total Cases: {total_cases:,.0f} | States: {len(states)}</sup>",
            x=0.5,
            font=dict(size=20)
        ),
        height=1200,
        showlegend=True,
        legend=dict(x=0.85, y=0.98),
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Weekly Cases", row=1, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_xaxes(title_text="Total Cases", row=2, col=2)
    fig.update_xaxes(title_text="Month", row=3, col=1)
    fig.update_yaxes(title_text="Average Weekly Cases", row=3, col=1)
    
    # Save
    filename = disease.replace(' ', '_').lower()
    filepath = os.path.join(output_dir, f'{filename}_analysis.html')
    fig.write_html(filepath, include_plotlyjs='cdn')
    print(f"Saved {disease} analysis to: {filepath}")


def create_comparison_html(data, output_dir='visualizations'):
    """Create an interactive comparison of all diseases (normalized)."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Dark24
    
    for idx, disease in enumerate(sorted(data.keys())):
        national, _ = aggregate_disease_data(data[disease])
        weeks = sorted(national.keys())
        values = [national[w] for w in weeks]
        dates = [week_to_date(w) for w in weeks]
        
        # Normalize to [0, 1] for comparison
        max_val = max(values) if max(values) > 0 else 1
        normalized = [v / max_val for v in values]
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=normalized,
                mode='lines',
                name=disease,
                line=dict(color=colors[idx % len(colors)], width=1.5),
                visible='legendonly' if idx > 4 else True,
                hovertemplate=f"<b>{disease}</b><br>" +
                              "Date: %{x}<br>" +
                              "Normalized: %{y:.2%}<br>" +
                              f"(Max: {max_val:,.0f})<extra></extra>"
            )
        )
    
    fig.update_layout(
        title=dict(
            text="<b>Disease Comparison (Normalized)</b><br>" +
                 "<sup>Each disease normalized to its maximum value for relative comparison</sup>",
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title="Date",
        yaxis_title="Normalized Cases (% of peak)",
        height=700,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        yaxis=dict(tickformat='.0%'),
        hovermode='x unified'
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(count=10, label="10y", step="year", stepmode="backward"),
                dict(count=20, label="20y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    filepath = os.path.join(output_dir, 'disease_comparison.html')
    fig.write_html(filepath, include_plotlyjs='cdn')
    print(f"Saved comparison to: {filepath}")


def create_correlation_heatmap(data, output_dir='visualizations'):
    """Create correlation heatmap between diseases."""
    os.makedirs(output_dir, exist_ok=True)
    
    diseases = sorted(data.keys())
    
    # Get all unique weeks across all diseases
    all_weeks = set()
    disease_national = {}
    for disease in diseases:
        national, _ = aggregate_disease_data(data[disease])
        disease_national[disease] = national
        all_weeks.update(national.keys())
    
    all_weeks = sorted(all_weeks)
    
    # Create time-aligned matrix
    disease_matrix = []
    for disease in diseases:
        row = [disease_national[disease].get(w, 0) for w in all_weeks]
        disease_matrix.append(row)
    
    disease_matrix = np.array(disease_matrix)
    
    # Calculate correlation
    # Handle potential NaN from constant columns
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(disease_matrix)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=diseases,
        y=diseases,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation'),
        hovertemplate="Disease 1: %{y}<br>Disease 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Disease Temporal Correlation Matrix</b><br>" +
                 "<sup>Pearson correlation of weekly case counts across time</sup>",
            x=0.5,
            font=dict(size=18)
        ),
        height=700,
        width=800,
        xaxis=dict(tickangle=45),
        template='plotly_white'
    )
    
    filepath = os.path.join(output_dir, 'disease_correlation.html')
    fig.write_html(filepath, include_plotlyjs='cdn')
    print(f"Saved correlation heatmap to: {filepath}")


def create_index_html(output_dir='visualizations'):
    """Create an index HTML page linking to all visualizations."""
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tycho US Disease Dataset Visualizations</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f6fa;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .section {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }
        .card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        a {
            text-decoration: none;
            color: #2980b9;
            font-weight: 500;
        }
        a:hover {
            color: #3498db;
        }
        .description {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .overview-links {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 20px;
        }
        .overview-links a {
            background: #3498db;
            color: white;
            padding: 12px 24px;
            border-radius: 5px;
            font-size: 1.1em;
        }
        .overview-links a:hover {
            background: #2980b9;
        }
    </style>
</head>
<body>
    <h1>🦠 Tycho US Disease Dataset</h1>
    <p class="subtitle">Interactive visualizations of historical disease surveillance data</p>
    
    <div class="section">
        <h2>📊 Overview Visualizations</h2>
        <div class="overview-links">
            <a href="disease_overview.html">📈 All Diseases Overview</a>
            <a href="disease_summary_table.html">📋 Summary Table</a>
            <a href="disease_comparison.html">🔄 Normalized Comparison</a>
            <a href="disease_correlation.html">🔗 Correlation Matrix</a>
        </div>
    </div>
    
    <div class="section">
        <h2>🔬 Individual Disease Analysis</h2>
        <div class="grid" id="disease-grid">
            <!-- Disease cards will be listed here -->
        </div>
    </div>
    
    <script>
        const diseases = DISEASE_LIST_PLACEHOLDER;
        const grid = document.getElementById('disease-grid');
        diseases.forEach(disease => {
            const card = document.createElement('div');
            card.className = 'card';
            const filename = disease.toLowerCase().replace(/ /g, '_');
            card.innerHTML = `
                <a href="${filename}_analysis.html">${disease}</a>
                <p class="description">Detailed time series, state heatmap, and seasonal analysis</p>
            `;
            grid.appendChild(card);
        });
    </script>
</body>
</html>
"""
    
    # Get list of diseases
    data = load_tycho_data()
    diseases = sorted(data.keys())
    
    # Replace placeholder with actual disease list
    html_content = html_content.replace('DISEASE_LIST_PLACEHOLDER', str(diseases))
    
    filepath = os.path.join(output_dir, 'index.html')
    with open(filepath, 'w') as f:
        f.write(html_content)
    print(f"Saved index to: {filepath}")


def main():
    """Main function to generate all visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate interactive visualizations for Tycho US disease data')
    parser.add_argument('--data', type=str, default='tycho_US.pt', help='Path to data file')
    parser.add_argument('--output', type=str, default='visualizations', help='Output directory')
    parser.add_argument('--disease', type=str, default=None, help='Generate only for specific disease')
    args = parser.parse_args()
    
    # Load data
    data = load_tycho_data(args.data)
    print(f"\nFound {len(data)} diseases: {sorted(data.keys())}\n")
    
    output_dir = args.output
    if not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_dir)
    
    if args.disease:
        # Generate for single disease
        create_single_disease_html(data, args.disease, output_dir)
    else:
        # Generate all visualizations
        print("=" * 60)
        print("Generating overview visualizations...")
        print("=" * 60)
        create_disease_overview_html(data, output_dir)
        
        print("\n" + "=" * 60)
        print("Generating comparison visualization...")
        print("=" * 60)
        create_comparison_html(data, output_dir)
        
        print("\n" + "=" * 60)
        print("Generating correlation heatmap...")
        print("=" * 60)
        create_correlation_heatmap(data, output_dir)
        
        print("\n" + "=" * 60)
        print("Generating individual disease analyses...")
        print("=" * 60)
        for disease in sorted(data.keys()):
            create_single_disease_html(data, disease, output_dir)
        
        print("\n" + "=" * 60)
        print("Generating index page...")
        print("=" * 60)
        create_index_html(output_dir)
        
        print("\n" + "=" * 60)
        print(f"✅ All visualizations saved to: {output_dir}")
        print(f"   Open {os.path.join(output_dir, 'index.html')} in a browser to explore!")
        print("=" * 60)


if __name__ == '__main__':
    main()
