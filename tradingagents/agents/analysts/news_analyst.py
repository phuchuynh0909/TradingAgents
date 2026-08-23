from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
)


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        # Must match the "news" ToolNode in graph/trading_graph.py: a bound tool
        # the node cannot execute turns into a failed call, and the model then
        # reports the data as unavailable.
        tools = [
            get_news,
            get_global_news,
            get_macro_indicators,
        ]

        system_message = (
            f"You are a news researcher covering the Vietnamese stock market (HOSE, HNX, "
            f"UPCoM). Analyse the news and macro releases of the past week and write a "
            f"comprehensive report on the state of the world as it bears on trading this "
            f"{asset_label} and on Vietnamese macroeconomics.\n\n"
            "TOOLS\n"
            f"- get_news(ticker, start_date, end_date): news for this {asset_label} by ticker, "
            "together with its industry and sector context (sector metrics and sector research).\n"
            "- get_global_news(curr_date, look_back_days, limit): the macro backdrop — the "
            "Vietnamese market and economy, plus the global picture (Fed policy, the dollar, "
            "US yields, China demand, commodities) that reaches Vietnam through the exchange "
            "rate, foreign flows and export orders. This is also where US and global figures "
            "come from.\n"
            "- get_macro_indicators(indicator, curr_date, look_back_days): dated Vietnamese "
            "macro releases on one topic. The feed is VIETNAM-ONLY. Ask for: 'monetary_policy' "
            "or 'sbv' (State Bank of Vietnam policy, open-market operations, system liquidity, "
            "money supply), 'interest_rate' (policy, deposit, lending and interbank rates, "
            "government bond yields), 'inflation' or 'cpi' (consumer prices), 'fx' or 'usdvnd' "
            "(the exchange rate), 'gdp' (growth), 'pmi' or 'industrial_production' "
            "(manufacturing), 'fdi' (registered and disbursed investment), 'trade', 'exports' "
            "or 'trade_balance', 'retail_sales' (domestic consumption), and 'credit' or "
            "'banking' (credit growth and the banking system). A US series — 'core_pce', "
            "'unemployment', '10y_treasury', 'fed_funds_rate' — is not covered: the tool "
            "answers with the general Vietnamese digest and says the series is missing. Take "
            "that data from get_global_news instead.\n\n"
            "WHAT MOVES THIS MARKET\n"
            "Weigh, where the evidence supports it: SBV MONETARY POLICY (policy rates, "
            "open-market operations and bill issuance, system liquidity, the credit-growth "
            "quota); INTEREST RATES (deposit and lending rates, interbank overnight, "
            "government bond yields); INFLATION (headline and core CPI against the year's "
            "target); the USD/VND EXCHANGE RATE and SBV intervention, which drive foreign "
            "flows; foreign investors' net buying or selling and the level of margin lending; "
            "GDP growth, PMI, industrial production, retail sales, FDI, exports and the trade "
            "balance; the public-investment, real-estate and corporate-bond cycles; and "
            "market-structure news (listing and settlement rules, FTSE Russell "
            "emerging-market reclassification). Connect each one to this ticker and its "
            "sector — transmission, not a list.\n\n"
            "DISCIPLINE\n"
            "Cite concrete headlines with their dates and attribute every figure to the tool "
            "that returned it. Knowledge-base excerpts are undated research: treat them as "
            "background rather than breaking news. When a tool reports that data is "
            "unavailable, say so in the report — do not substitute a remembered or estimated "
            "number, and never carry a US figure over to Vietnam.\n\n"
            "Provide specific, actionable insights with supporting evidence to help traders "
            "make informed decisions. "
            "Make sure to append a Markdown table at the end of the report to organize key "
            "points in the report, organized and easy to read."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
