from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_macro_indicators(
    indicator: Annotated[
        str,
        "Macro topic: 'monetary_policy' or 'sbv', 'interest_rate', 'inflation' "
        "or 'cpi', 'fx' or 'usdvnd', 'gdp', 'pmi' or 'industrial_production', "
        "'fdi', 'trade' / 'exports' / 'trade_balance', 'retail_sales', 'credit' "
        "or 'banking'.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format; the end of the window"],
    look_back_days: Annotated[
        int | None, "Trailing window length in days; omit for a 30-day window"
    ] = None,
) -> str:
    """
    Retrieve dated Vietnamese macroeconomic releases for one topic: State Bank of
    Vietnam policy and liquidity, interest rates, inflation, the USD/VND exchange
    rate, growth, manufacturing, investment, trade, consumption and bank credit.
    Returns the releases in the window, most recent first, each with its date and
    a summary. Uses the configured macro_data vendor.

    The feed covers Vietnam only. A US series ('core_pce', 'unemployment',
    '10y_treasury', 'fed_funds_rate') returns the general Vietnamese digest with a
    note that the series is not covered — take US and global data from
    get_global_news instead.

    Args:
        indicator (str): Macro topic, as listed above
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Trailing window length; omit for a 30-day window

    Returns:
        str: A formatted markdown report of the releases on that topic
    """
    return route_to_vendor("get_macro_indicators", indicator, curr_date, look_back_days)
