import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Net Worth Forecast", layout="wide")

st.title("📈 Net Worth Forecast Simulator")
st.markdown("Project your financial future with retirement planning and debt payoff")

# Sidebar inputs
st.sidebar.header("Personal Info")
current_age = st.sidebar.slider("Current Age", 20, 60, 30)
retirement_age = st.sidebar.slider("Retirement Age", 55, 75, 65)
current_income = st.sidebar.slider("Current Income ($)", 40000, 300000, 75000, 5000)
raise_rate = st.sidebar.slider("Annual Raise (%)", 0.0, 10.0, 3.0, 0.5)

st.sidebar.header("Savings & Debt")
current_savings = st.sidebar.slider("Current Retirement Savings ($)", 0, 500000, 25000, 5000)
current_debt = st.sidebar.slider("Current Debt ($)", 0, 500000, 200000, 5000)

# SEPARATED: Required payment vs extra principal
st.sidebar.subheader("💳 Debt Payments")
debt_payment = st.sidebar.slider("Required Monthly Payment ($)", 500, 5000, 1500, 50)
extra_principal = st.sidebar.slider("Extra Principal Payment ($)", 0, 3000, 0, 100)
debt_rate = st.sidebar.slider("Debt Interest Rate (%)", 2.0, 8.0, 4.0, 0.125)

# NEW: Post debt-free investment strategy
st.sidebar.subheader("💰 After Debt-Free")
redirect_debt_payment = st.sidebar.checkbox("Redirect debt payment to investments after payoff", value=True)
redirect_extra_principal = st.sidebar.checkbox("Redirect extra principal to investments after payoff", value=True)
additional_post_debt = st.sidebar.slider("Additional Monthly Investment After Debt-Free ($)", 0, 2000, 0, 100)

st.sidebar.header("Investment Strategy")
contribution_rate = st.sidebar.slider("Your Contribution (% of income)", 0, 30, 15, 1)
employer_match = st.sidebar.slider("Employer Match (% of income)", 0.0, 10.0, 5.0, 0.5)
return_rate = st.sidebar.slider("Expected Return (%)", 3.0, 12.0, 8.0, 0.5)
volatility = st.sidebar.slider("Volatility (%)", 5, 30, 15, 1)

st.sidebar.header("Monte Carlo Simulation")
run_monte_carlo = st.sidebar.checkbox("Run Monte Carlo Simulation")
num_simulations = st.sidebar.slider("Number of Simulations", 50, 500, 100, 50) if run_monte_carlo else 100


def simulate_trajectory(current_age, retirement_age, current_income, current_savings, 
                       current_debt, debt_payment, extra_principal, debt_rate, 
                       contribution_rate, employer_match, return_rate, raise_rate,
                       redirect_debt_payment, redirect_extra_principal, additional_post_debt,
                       use_stochastic=False, seed=None):
    """Simulate financial trajectory with separate debt payment tracking"""
    
    if seed is not None:
        np.random.seed(seed)
    
    years = retirement_age - current_age
    months = years * 12
    
    income = current_income
    retirement_balance = current_savings
    debt_balance = current_debt
    monthly_debt_rate = debt_rate / 100 / 12
    
    # Track when debt becomes free
    is_debt_free = current_debt == 0
    
    ages = []
    net_worths = []
    retirement_balances = []
    debt_balances = []
    incomes = []
    monthly_investments = []
    total_debt_payments = []
    
    for m in range(1, months + 1):
        age = current_age + (m / 12)
        
        # Annual raise
        if m % 12 == 1 and m > 1:
            income *= (1 + raise_rate / 100)
        
        monthly_income = income / 12
        
        # Base retirement contributions
        employee_contrib = monthly_income * (contribution_rate / 100)
        employer_contrib = monthly_income * (employer_match / 100)
        total_contrib = employee_contrib + employer_contrib
        
        # Track total debt payment this month
        total_debt_payment_this_month = 0
        
        # Debt paydown
        if debt_balance > 0:
            interest = debt_balance * monthly_debt_rate
            
            # Required payment
            principal = min(debt_payment - interest, debt_balance)
            debt_balance -= principal
            total_debt_payment_this_month += debt_payment
            
            # Extra principal payment
            if extra_principal > 0 and debt_balance > 0:
                extra_applied = min(extra_principal, debt_balance)
                debt_balance -= extra_applied
                total_debt_payment_this_month += extra_applied
            
            debt_balance = max(0, debt_balance)
            
            # Check if we just became debt-free
            if debt_balance <= 0 and not is_debt_free:
                is_debt_free = True
        
        # Post debt-free: redirect payments to investments
        if is_debt_free and current_debt > 0:  # Only redirect if there WAS debt
            if redirect_debt_payment:
                total_contrib += debt_payment
            if redirect_extra_principal:
                total_contrib += extra_principal
            if additional_post_debt > 0:
                total_contrib += additional_post_debt
        
        # Investment returns
        monthly_return = return_rate / 100 / 12
        if use_stochastic:
            monthly_vol = (volatility / 100) / np.sqrt(12)
            shock = np.random.randn() * monthly_vol
            monthly_return += shock
        
        retirement_balance *= (1 + monthly_return)
        retirement_balance += total_contrib
        
        net_worth = retirement_balance - debt_balance
        
        # Sample quarterly
        if m % 3 == 0:
            ages.append(age)
            net_worths.append(net_worth)
            retirement_balances.append(retirement_balance)
            debt_balances.append(debt_balance)
            incomes.append(income)
            monthly_investments.append(total_contrib)
            total_debt_payments.append(total_debt_payment_this_month)
    
    return pd.DataFrame({
        'age': ages,
        'net_worth': net_worths,
        'retirement': retirement_balances,
        'debt': debt_balances,
        'income': incomes,
        'monthly_investment': monthly_investments,
        'debt_payment': total_debt_payments
    })


# Run deterministic simulation
df = simulate_trajectory(
    current_age, retirement_age, current_income, current_savings,
    current_debt, debt_payment, extra_principal, debt_rate, contribution_rate,
    employer_match, return_rate, raise_rate, redirect_debt_payment, 
    redirect_extra_principal, additional_post_debt, use_stochastic=False
)

# Calculate key milestones
final_net_worth = df['net_worth'].iloc[-1]
final_retirement = df['retirement'].iloc[-1]
debt_free_age = df[df['debt'] <= 0]['age'].iloc[0] if len(df[df['debt'] <= 0]) > 0 else None
millionaire_age = df[df['net_worth'] >= 1_000_000]['age'].iloc[0] if len(df[df['net_worth'] >= 1_000_000]) > 0 else None

# Calculate how much gets redirected after debt-free
redirected_amount = 0
if redirect_debt_payment:
    redirected_amount += debt_payment
if redirect_extra_principal:
    redirected_amount += extra_principal
if additional_post_debt > 0:
    redirected_amount += additional_post_debt

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Final Net Worth",
        f"${final_net_worth/1e6:.2f}M" if final_net_worth >= 1e6 else f"${final_net_worth:,.0f}",
        help=f"At age {retirement_age}"
    )

with col2:
    st.metric(
        "Retirement Portfolio",
        f"${final_retirement/1e6:.2f}M" if final_retirement >= 1e6 else f"${final_retirement:,.0f}",
        help=f"At age {retirement_age}"
    )

with col3:
    if debt_free_age:
        years_to_debt_free = int(debt_free_age - current_age)
        st.metric(
            "Debt-Free Age",
            f"{int(debt_free_age)}",
            delta=f"In {years_to_debt_free} years",
            delta_color="normal"
        )
    else:
        st.metric("Debt-Free Age", "Already debt-free!")

with col4:
    if millionaire_age:
        years_to_millionaire = int(millionaire_age - current_age)
        st.metric(
            "Millionaire Age",
            f"{int(millionaire_age)}",
            delta=f"In {years_to_millionaire} years",
            delta_color="normal"
        )
    else:
        st.metric("Millionaire Age", "Not reached")

# Show debt payment breakdown
if current_debt > 0:
    st.info(f"""
    **💳 Debt Payment Strategy:**
    - Required Payment: ${debt_payment:,.0f}/month
    - Extra Principal: ${extra_principal:,.0f}/month
    - **Total Debt Payment: ${debt_payment + extra_principal:,.0f}/month**
    
    **💰 After Debt-Free (redirected to investments):**
    - From debt payment: ${debt_payment:,.0f}/month {'✓' if redirect_debt_payment else '✗'}
    - From extra principal: ${extra_principal:,.0f}/month {'✓' if redirect_extra_principal else '✗'}
    - Additional investment: ${additional_post_debt:,.0f}/month
    - **Total redirected: ${redirected_amount:,.0f}/month**
    """)

# Monte Carlo simulation
percentiles_df = None
if run_monte_carlo:
    with st.spinner(f'Running {num_simulations} Monte Carlo simulations...'):
        all_trajectories = []
        for i in range(num_simulations):
            sim_df = simulate_trajectory(
                current_age, retirement_age, current_income, current_savings,
                current_debt, debt_payment, extra_principal, debt_rate, contribution_rate,
                employer_match, return_rate, raise_rate, redirect_debt_payment,
                redirect_extra_principal, additional_post_debt, use_stochastic=True, seed=i
            )
            all_trajectories.append(sim_df)
        
        # Calculate percentiles
        ages = df['age'].values
        percentiles = {
            'age': ages,
            'p10': [],
            'p25': [],
            'p50': [],
            'p75': [],
            'p90': []
        }
        
        for i in range(len(ages)):
            values = [traj['net_worth'].iloc[i] for traj in all_trajectories]
            percentiles['p10'].append(np.percentile(values, 10))
            percentiles['p25'].append(np.percentile(values, 25))
            percentiles['p50'].append(np.percentile(values, 50))
            percentiles['p75'].append(np.percentile(values, 75))
            percentiles['p90'].append(np.percentile(values, 90))
        
        percentiles_df = pd.DataFrame(percentiles)

# Create visualizations
st.subheader("📊 Financial Projections")

# Create main chart with 3 rows
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=("Net Worth Projection", "Retirement Portfolio Growth",
                   "Debt Payoff Timeline", "Income Progression",
                   "Monthly Investment Contributions", "Total Monthly Cash Flow"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}]]
)

# Net worth with Monte Carlo if enabled
if run_monte_carlo and percentiles_df is not None:
    # Add uncertainty bands
    fig.add_trace(
        go.Scatter(
            x=percentiles_df['age'], 
            y=percentiles_df['p90'],
            mode='lines',
            name='90th percentile',
            line=dict(width=0),
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=percentiles_df['age'], 
            y=percentiles_df['p10'],
            mode='lines',
            name='10th percentile',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty',
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=percentiles_df['age'], 
            y=percentiles_df['p50'],
            mode='lines',
            name='Median',
            line=dict(color='blue', width=3),
            showlegend=True
        ),
        row=1, col=1
    )

# Deterministic line
fig.add_trace(
    go.Scatter(
        x=df['age'], 
        y=df['net_worth'],
        mode='lines',
        name='Expected',
        line=dict(color='green', width=2, dash='dash'),
        showlegend=True
    ),
    row=1, col=1
)

# Retirement balance
fig.add_trace(
    go.Scatter(x=df['age'], y=df['retirement'], 
               name="Portfolio", line=dict(color='green', width=2),
               fill='tozeroy', fillcolor='rgba(0,128,0,0.2)', showlegend=False),
    row=1, col=2
)

# Debt balance
fig.add_trace(
    go.Scatter(x=df['age'], y=df['debt'], 
               name="Debt", line=dict(color='red', width=2),
               fill='tozeroy', fillcolor='rgba(255,0,0,0.2)', showlegend=False),
    row=2, col=1
)

# Income
fig.add_trace(
    go.Scatter(x=df['age'], y=df['income'], 
               name="Income", line=dict(color='purple', width=2), showlegend=False),
    row=2, col=2
)

# Monthly investment contributions (shows the boost after debt-free)
fig.add_trace(
    go.Scatter(x=df['age'], y=df['monthly_investment'], 
               name="Monthly Investment", line=dict(color='blue', width=2),
               fill='tozeroy', fillcolor='rgba(0,0,255,0.2)', showlegend=False),
    row=3, col=1
)

# Total monthly cash flow (investments + debt payments)
total_cash_flow = df['monthly_investment'] + df['debt_payment']
fig.add_trace(
    go.Scatter(x=df['age'], y=total_cash_flow, 
               name="Total Cash Flow", line=dict(color='orange', width=2),
               fill='tozeroy', fillcolor='rgba(255,165,0,0.2)', showlegend=False),
    row=3, col=2
)

# Update axes
fig.update_xaxes(title_text="Age", row=1, col=1)
fig.update_xaxes(title_text="Age", row=1, col=2)
fig.update_xaxes(title_text="Age", row=2, col=1)
fig.update_xaxes(title_text="Age", row=2, col=2)
fig.update_xaxes(title_text="Age", row=3, col=1)
fig.update_xaxes(title_text="Age", row=3, col=2)

fig.update_yaxes(title_text="Net Worth ($)", row=1, col=1)
fig.update_yaxes(title_text="Portfolio ($)", row=1, col=2)
fig.update_yaxes(title_text="Debt ($)", row=2, col=1)
fig.update_yaxes(title_text="Income ($)", row=2, col=2)
fig.update_yaxes(title_text="Investment ($)", row=3, col=1)
fig.update_yaxes(title_text="Cash Flow ($)", row=3, col=2)

fig.update_layout(height=1000, showlegend=True)

st.plotly_chart(fig, use_container_width=True)

# Summary section
st.subheader("📋 Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Starting Position**")
    st.write(f"- Age: {current_age}")
    st.write(f"- Income: ${current_income:,.0f}")
    st.write(f"- Retirement Savings: ${current_savings:,.0f}")
    st.write(f"- Debt: ${current_debt:,.0f}")
    st.write(f"- Net Worth: ${current_savings - current_debt:,.0f}")
    st.write(f"- Monthly debt payment: ${debt_payment:,.0f}")
    st.write(f"- Monthly extra principal: ${extra_principal:,.0f}")

with col2:
    st.markdown("**At Retirement**")
    st.write(f"- Age: {retirement_age}")
    st.write(f"- Income: ${df['income'].iloc[-1]:,.0f}")
    st.write(f"- Retirement Portfolio: ${final_retirement:,.0f}")
    st.write(f"- Net Worth: ${final_net_worth:,.0f}")
    st.write(f"- Monthly investment: ${df['monthly_investment'].iloc[-1]:,.0f}")
    
st.markdown("---")
    
# Key insights
st.markdown("**💡 Key Insights:**")
if debt_free_age:
    st.write(f"- You'll be **debt-free at age {int(debt_free_age)}** ({int(debt_free_age - current_age)} years from now)")
    if redirected_amount > 0:
        years_investing_extra = retirement_age - debt_free_age
        st.write(f"- After debt-free, you'll invest an **extra ${redirected_amount:,.0f}/month** for {years_investing_extra:.1f} years")
if millionaire_age:
    st.write(f"- You'll become a **millionaire at age {int(millionaire_age)}** ({int(millionaire_age - current_age)} years from now)")
st.write(f"- Your retirement portfolio will grow to **${final_retirement:,.0f}** by age {retirement_age}")
st.write(f"- Total contributions over career: **{contribution_rate + employer_match}%** of income ({contribution_rate}% you + {employer_match}% employer)")

# Show impact of debt payoff strategy
if debt_free_age and current_debt > 0:
    pre_debt_free_investment = df[df['age'] < debt_free_age]['monthly_investment'].iloc[-1] if len(df[df['age'] < debt_free_age]) > 0 else 0
    post_debt_free_investment = df[df['age'] >= debt_free_age]['monthly_investment'].iloc[0] if len(df[df['age'] >= debt_free_age]) > 0 else 0
    boost = post_debt_free_investment - pre_debt_free_investment
    if boost > 0:
        st.write(f"- Your monthly investments will **increase by ${boost:,.0f}** once debt is paid off!")
