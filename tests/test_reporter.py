import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import shutil
import tempfile
from core.reporter import Reporter

class TestReporter:
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def reporter(self, temp_dir):
        return Reporter(output_dir=temp_dir)

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2023-01-01', end='2023-01-31')
        equity_df = pd.DataFrame({
            'TotalValue': np.linspace(10000, 11000, len(dates))
        }, index=dates)
        equity_df.index.name = 'Date'
        
        trades_df = pd.DataFrame({
            'Date': dates[:5],
            'Symbol': ['TEST'] * 5,
            'Action': ['BUY'] * 5,
            'Price': [100] * 5,
            'Quantity': [10] * 5,
            'Value': [100] * 5
        })
        
        metrics = {
            'Total Return': 0.1,
            'Sharpe Ratio': 1.5,
            'Max Drawdown': -0.05
        }
        
        return equity_df, trades_df, metrics

    def test_init(self, temp_dir):
        reporter = Reporter(output_dir=temp_dir)
        assert Path(temp_dir).exists()

    def test_init_default(self):
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            reporter = Reporter()
            assert reporter.output_dir == Path('reports')
            mock_mkdir.assert_called()

    @patch('core.reporter.Reporter._generate_basic_html')
    def test_generate_html_report_no_plotly(self, mock_basic_html, reporter, sample_data):
        equity_df, trades_df, metrics = sample_data
        
        # Simulate ImportError for plotly
        with patch.dict('sys.modules', {'plotly': None}):
            reporter.generate_html_report(equity_df, trades_df, metrics)
            mock_basic_html.assert_called_once()

    def test_generate_html_report_with_plotly(self, reporter, sample_data):
        equity_df, trades_df, metrics = sample_data
        
        # Mock plotly modules
        with patch('plotly.graph_objects.Scatter') as mock_scatter, \
             patch('plotly.graph_objects.Histogram') as mock_hist, \
             patch('plotly.graph_objects.Heatmap') as mock_heatmap, \
             patch('plotly.subplots.make_subplots') as mock_subplots, \
             patch('plotly.io.to_html') as mock_to_html:
            
            mock_fig = MagicMock()
            mock_subplots.return_value = mock_fig
            mock_to_html.return_value = "<div>Plotly Chart</div>"
            
            html = reporter.generate_html_report(equity_df, trades_df, metrics)
            
            assert isinstance(html, str)
            assert "<div>Plotly Chart</div>" in html
            assert "<!DOCTYPE html>" in html
            mock_subplots.assert_called()

    def test_generate_html_report_save_file(self, reporter, sample_data):
        equity_df, trades_df, metrics = sample_data
        save_path = Path(reporter.output_dir) / "report.html"
        
        with patch('plotly.graph_objects.Scatter'), \
             patch('plotly.subplots.make_subplots') as mock_subplots, \
             patch('plotly.io.to_html') as mock_to_html:
            
            mock_fig = MagicMock()
            mock_subplots.return_value = mock_fig
            mock_to_html.return_value = "<div>Plotly Chart</div>"
            
            reporter.generate_html_report(equity_df, trades_df, metrics, save_path=str(save_path))
            
            assert save_path.exists()
            content = save_path.read_text()
            assert "<div>Plotly Chart</div>" in content

    def test_generate_html_report_with_benchmark(self, reporter, sample_data):
        equity_df, trades_df, metrics = sample_data
        benchmark_df = equity_df.copy()
        benchmark_df['TotalValue'] = benchmark_df['TotalValue'] * 0.9
        
        with patch('plotly.graph_objects.Scatter'), \
             patch('plotly.subplots.make_subplots') as mock_subplots, \
             patch('plotly.io.to_html') as mock_to_html, \
             patch('core.benchmark.BenchmarkComparator') as mock_comparator:
            
            mock_fig = MagicMock()
            mock_subplots.return_value = mock_fig
            mock_to_html.return_value = "<div>Plotly Chart</div>"
            
            # Mock benchmark comparator
            mock_comp_instance = mock_comparator.return_value
            mock_comp_instance.calculate_metrics.return_value = {'rolling_beta_90d': pd.Series([1.0]*len(equity_df), index=equity_df.index)}
            mock_comp_instance.format_for_base_100.return_value = (equity_df, benchmark_df)
            
            reporter.generate_html_report(equity_df, trades_df, metrics, benchmark_df=benchmark_df)
            
            # Verify benchmark related calls
            mock_comparator.assert_called()
            mock_comp_instance.calculate_metrics.assert_called()

    def test_calculate_monthly_returns(self, reporter):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31')
        returns = pd.Series(0.01, index=dates) # 1% daily return
        
        monthly = reporter._calculate_monthly_returns(returns)
        
        assert isinstance(monthly, pd.DataFrame)
        assert 2023 in monthly.index
        assert 'Jan' in monthly.columns
        assert 'Dec' in monthly.columns

    def test_metrics_to_html(self, reporter):
        metrics = {'Return': 0.1, 'Sharpe': 1.5}
        html = reporter._metrics_to_html(metrics)
        assert '<table' in html
        assert 'Return' in html
        assert '10.00%' in html # 0.1 formatted as percentage

    def test_trades_summary_to_html(self, reporter, sample_data):
        _, trades_df, _ = sample_data
        html = reporter._trades_summary_to_html(trades_df)
        assert '<table' in html
        assert 'TEST' in html

    def test_worst_days_to_html(self, reporter, sample_data):
        equity_df, _, _ = sample_data
        returns = equity_df['TotalValue'].pct_change().fillna(0)
        html = reporter._worst_days_to_html(equity_df, returns)
        assert '<table' in html
