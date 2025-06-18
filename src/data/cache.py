"""
缓存模块：用于在内存中缓存API响应数据
此模块提供了一个Cache类，用于存储和管理各种金融数据的内存缓存，包括：
- 股票价格数据
- 财务指标
- 财务报表项目
- 内部交易信息
- 公司新闻
"""

class Cache:
    """
    内存缓存类：用于缓存API响应数据
    提供了对不同类型金融数据的缓存管理功能
    """

    def __init__(self):
        """
        初始化缓存对象
        创建用于存储不同类型数据的缓存字典
        """
        self._prices_cache: dict[str, list[dict[str, any]]] = {}  # 股票价格缓存
        self._financial_metrics_cache: dict[str, list[dict[str, any]]] = {}  # 财务指标缓存
        self._line_items_cache: dict[str, list[dict[str, any]]] = {}  # 财务报表项目缓存
        self._insider_trades_cache: dict[str, list[dict[str, any]]] = {}  # 内部交易缓存
        self._company_news_cache: dict[str, list[dict[str, any]]] = {}  # 公司新闻缓存

    def _merge_data(self, existing: list[dict] | None, new_data: list[dict], key_field: str) -> list[dict]:
        """
        合并现有数据和新数据，基于指定的键字段避免重复
        
        参数:
            existing: 现有的数据列表
            new_data: 新的数据列表
            key_field: 用于判断重复的键字段名
        
        返回:
            合并后的数据列表
        """
        if not existing:
            return new_data

        # Create a set of existing keys for O(1) lookup
        existing_keys = {item[key_field] for item in existing}

        # Only add items that don't exist yet
        merged = existing.copy()
        merged.extend([item for item in new_data if item[key_field] not in existing_keys])
        return merged

    def get_prices(self, ticker: str) -> list[dict[str, any]] | None:
        """
        获取缓存的股票价格数据
        
        参数:
            ticker: 股票代码
        
        返回:
            该股票的价格数据列表，如果不存在则返回None
        """
        return self._prices_cache.get(ticker)

    def set_prices(self, ticker: str, data: list[dict[str, any]]):
        """
        设置股票价格数据到缓存
        
        参数:
            ticker: 股票代码
            data: 要缓存的价格数据列表
        """
        self._prices_cache[ticker] = self._merge_data(self._prices_cache.get(ticker), data, key_field="time")

    def get_financial_metrics(self, ticker: str) -> list[dict[str, any]]:
        """
        获取缓存的财务指标数据
        
        参数:
            ticker: 股票代码
        
        返回:
            该股票的财务指标数据列表
        """
        return self._financial_metrics_cache.get(ticker)

    def set_financial_metrics(self, ticker: str, data: list[dict[str, any]]):
        """
        设置财务指标数据到缓存
        
        参数:
            ticker: 股票代码
            data: 要缓存的财务指标数据列表
        """
        self._financial_metrics_cache[ticker] = self._merge_data(self._financial_metrics_cache.get(ticker), data, key_field="report_period")

    def get_line_items(self, ticker: str) -> list[dict[str, any]] | None:
        """
        获取缓存的财务报表项目数据
        
        参数:
            ticker: 股票代码
        
        返回:
            该股票的财务报表项目数据列表，如果不存在则返回None
        """
        return self._line_items_cache.get(ticker)

    def set_line_items(self, ticker: str, data: list[dict[str, any]]):
        """
        设置财务报表项目数据到缓存
        
        参数:
            ticker: 股票代码
            data: 要缓存的财务报表项目数据列表
        """
        self._line_items_cache[ticker] = self._merge_data(self._line_items_cache.get(ticker), data, key_field="report_period")

    def get_insider_trades(self, ticker: str) -> list[dict[str, any]] | None:
        """
        获取缓存的内部交易数据
        
        参数:
            ticker: 股票代码
        
        返回:
            该股票的内部交易数据列表，如果不存在则返回None
        """
        return self._insider_trades_cache.get(ticker)

    def set_insider_trades(self, ticker: str, data: list[dict[str, any]]):
        """
        设置内部交易数据到缓存
        
        参数:
            ticker: 股票代码
            data: 要缓存的内部交易数据列表
        """
        self._insider_trades_cache[ticker] = self._merge_data(self._insider_trades_cache.get(ticker), data, key_field="filing_date")  # Could also use transaction_date if preferred

    def get_company_news(self, ticker: str) -> list[dict[str, any]] | None:
        """
        获取缓存的公司新闻数据
        
        参数:
            ticker: 股票代码
        
        返回:
            该股票的公司新闻数据列表，如果不存在则返回None
        """
        return self._company_news_cache.get(ticker)

    def set_company_news(self, ticker: str, data: list[dict[str, any]]):
        """
        设置公司新闻数据到缓存
        
        参数:
            ticker: 股票代码
            data: 要缓存的公司新闻数据列表
        """
        self._company_news_cache[ticker] = self._merge_data(self._company_news_cache.get(ticker), data, key_field="date")


# 全局缓存实例
_cache = Cache()


def get_cache() -> Cache:
    """
    获取全局缓存实例
    
    返回:
        全局Cache对象实例
    """
    return _cache
