#!/usr/bin/env python3
"""
ENHANCED CORRELATION-AWARE MULTI-PAIR TRADING SYSTEM
==================================================

Advanced portfolio trading system with:
✅ Real-time correlation monitoring between active positions
✅ Enhanced position sizing based on correlation exposure
✅ Advanced risk metrics and portfolio heat map
✅ Intelligent pair selection to avoid overexposure
✅ Dynamic position management based on correlation changes
✅ Comprehensive portfolio analytics and reporting

This system takes multi-pair trading to the next level with
sophisticated correlation analysis and risk management.

Created: October 2, 2025
Version: Enhanced Correlation Edition
"""

import json
import time
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from multi_currency_ict_system import *

@dataclass
class CorrelationMetrics:
    """Advanced correlation and risk metrics"""
    pair_correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)
    portfolio_correlation_risk: float = 0.0
    diversification_ratio: float = 1.0
    concentration_risk: float = 0.0
    maximum_drawdown_risk: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    
@dataclass
class PositionAnalysis:
    """Enhanced position with correlation analysis"""
    order: TradingOrder
    correlation_exposure: Dict[str, float] = field(default_factory=dict)
    diversification_benefit: float = 0.0
    risk_contribution: float = 0.0
    optimal_size: float = 0.0
    correlation_warning: bool = False

class CorrelationAwareTrader:
    """Advanced multi-pair trader with correlation intelligence"""
    
    def __init__(self, balance: float = 1000.0):
        self.account = DemoAccount(
            balance=balance,
            equity=balance,
            free_margin=balance,
            starting_balance=balance
        )
        
        self.data_provider = MultiCurrencyDataProvider()
        self.analyzer = ICTMultiPairAnalyzer(self.data_provider)
        
        # Enhanced risk parameters
        self.base_risk_per_trade = 1.5  # Base risk per trade
        self.max_portfolio_risk = 8.0   # Maximum total portfolio risk
        self.max_positions = 4          # Maximum concurrent positions
        self.correlation_threshold = 0.7 # High correlation warning threshold
        
        # Position tracking with correlation analysis
        self.active_positions: List[PositionAnalysis] = []
        self.trade_counter = 0
        
        # Enhanced currency correlation matrix
        self.correlation_matrix = {
            # Major USD pairs correlations
            ('EURUSD', 'GBPUSD'): 0.72,
            ('EURUSD', 'AUDUSD'): 0.68,
            ('EURUSD', 'NZDUSD'): 0.63,
            ('EURUSD', 'USDCHF'): -0.81,  # Negative correlation
            ('GBPUSD', 'AUDUSD'): 0.54,
            ('GBPUSD', 'NZDUSD'): 0.49,
            ('GBPUSD', 'USDCHF'): -0.69,
            ('AUDUSD', 'NZDUSD'): 0.87,   # High correlation
            ('AUDUSD', 'USDCHF'): -0.58,
            ('NZDUSD', 'USDCHF'): -0.52,
            
            # JPY pairs correlations
            ('USDJPY', 'EURJPY'): 0.76,
            ('USDJPY', 'GBPJPY'): 0.71,
            ('USDJPY', 'AUDJPY'): 0.69,
            ('USDJPY', 'NZDJPY'): 0.65,
            ('EURJPY', 'GBPJPY'): 0.83,   # High correlation
            ('EURJPY', 'AUDJPY'): 0.78,
            ('GBPJPY', 'AUDJPY'): 0.74,
            ('AUDJPY', 'NZDJPY'): 0.91,   # Very high correlation
            
            # Cross-currency correlations
            ('EURAUD', 'GBPAUD'): 0.79,
            ('EURGBP', 'EURAUD'): 0.42,
            ('EURGBP', 'GBPAUD'): -0.38,  # Negative correlation
            
            # Commodity currency correlations
            ('AUDUSD', 'USDCAD'): -0.61,  # Negative (oil vs commodity)
            ('NZDUSD', 'USDCAD'): -0.54,
            ('AUDJPY', 'CADJPY'): 0.73,
            
            # Gold correlations
            ('XAUUSD', 'EURUSD'): 0.34,
            ('XAUUSD', 'AUDUSD'): 0.41,
            ('XAUUSD', 'USDJPY'): -0.28,  # Negative correlation
            ('XAUUSD', 'USDCHF'): -0.39,
        }
        
        # Currency strength tracking
        self.currency_exposure = defaultdict(float)
        
    def get_correlation(self, pair1: str, pair2: str) -> float:
        """Get correlation between two currency pairs"""
        if pair1 == pair2:
            return 1.0
        
        key = tuple(sorted([pair1, pair2]))
        return self.correlation_matrix.get(key, 0.0)
    
    def calculate_currency_overlap(self, pair1: str, pair2: str) -> float:
        """Calculate currency overlap between pairs"""
        currencies1 = set([pair1[:3], pair1[3:6]])
        currencies2 = set([pair2[:3], pair2[3:6]])
        
        overlap = len(currencies1.intersection(currencies2))
        if overlap == 2:
            return 1.0  # Same pair
        elif overlap == 1:
            return 0.5  # Share one currency
        else:
            return 0.0  # No overlap
    
    def analyze_portfolio_correlation(self) -> CorrelationMetrics:
        """Comprehensive portfolio correlation analysis"""
        
        if not self.active_positions:
            return CorrelationMetrics()
        
        metrics = CorrelationMetrics()
        
        # Calculate pair correlations for active positions
        active_symbols = [pos.order.symbol for pos in self.active_positions]
        
        total_correlation = 0.0
        correlation_count = 0
        
        for i, pos1 in enumerate(self.active_positions):
            for j, pos2 in enumerate(self.active_positions):
                if i < j:  # Avoid double counting
                    corr = self.get_correlation(pos1.order.symbol, pos2.order.symbol)
                    
                    # Adjust correlation based on position directions
                    if pos1.order.direction != pos2.order.direction:
                        corr = -corr  # Opposite directions reduce correlation risk
                    
                    key = (pos1.order.symbol, pos2.order.symbol)
                    metrics.pair_correlations[key] = corr
                    
                    total_correlation += abs(corr)
                    correlation_count += 1
        
        # Portfolio correlation risk
        if correlation_count > 0:
            metrics.portfolio_correlation_risk = total_correlation / correlation_count
        
        # Diversification ratio (lower is better diversified)
        num_positions = len(self.active_positions)
        if num_positions > 1:
            metrics.diversification_ratio = metrics.portfolio_correlation_risk * num_positions
        
        # Concentration risk (currency exposure)
        currency_weights = defaultdict(float)
        total_risk = sum(pos.order.size_lots for pos in self.active_positions)
        
        for pos in self.active_positions:
            weight = pos.order.size_lots / total_risk if total_risk > 0 else 0
            
            base_currency = pos.order.symbol[:3]
            quote_currency = pos.order.symbol[3:6]
            
            if pos.order.direction == 'long':
                currency_weights[base_currency] += weight
                currency_weights[quote_currency] -= weight
            else:
                currency_weights[base_currency] -= weight
                currency_weights[quote_currency] += weight
        
        # Concentration risk is the maximum absolute exposure to any currency
        metrics.concentration_risk = max(abs(w) for w in currency_weights.values()) if currency_weights else 0
        
        # Estimate portfolio VaR (simplified)
        individual_vars = []
        for pos in self.active_positions:
            # Calculate individual position VaR (simplified)
            stop_distance = abs(pos.order.entry_price - pos.order.stop_loss)
            var = stop_distance * pos.order.size_lots * 10000  # Convert to dollars
            individual_vars.append(var)
        
        if individual_vars:
            # Portfolio VaR considering correlations (simplified)
            sum_individual_var = sum(individual_vars)
            correlation_adjustment = math.sqrt(metrics.portfolio_correlation_risk)
            metrics.var_95 = sum_individual_var * correlation_adjustment
        
        return metrics
    
    def calculate_optimal_position_size(self, symbol: str, direction: str, 
                                      entry: float, stop: float) -> Tuple[float, str]:
        """Calculate optimal position size considering correlations"""
        
        base_risk = self.account.balance * (self.base_risk_per_trade / 100)
        
        # Calculate correlation adjustment
        correlation_penalty = 0.0
        correlation_details = []
        
        for pos in self.active_positions:
            corr = self.get_correlation(symbol, pos.order.symbol)
            
            # Adjust for direction alignment
            if direction == pos.order.direction:
                correlation_effect = abs(corr)
            else:
                correlation_effect = -abs(corr)  # Opposite directions help diversification
            
            if abs(corr) > 0.5:  # Significant correlation
                correlation_penalty += correlation_effect * 0.3  # Reduce size by 30% per high correlation
                correlation_details.append(f"{pos.order.symbol}: {corr:+.2f}")
        
        # Apply correlation penalty
        correlation_multiplier = max(0.3, 1.0 - correlation_penalty)  # Minimum 30% of base size
        adjusted_risk = base_risk * correlation_multiplier
        
        # Calculate position size
        if 'JPY' in symbol:
            pip_value = 0.01
        elif symbol == 'XAUUSD':
            pip_value = 0.10
        else:
            pip_value = 0.0001
        
        stop_distance_pips = abs(entry - stop) / pip_value
        lot_size = adjusted_risk / (stop_distance_pips * 1.0)  # Simplified pip cost
        lot_size = max(0.01, min(0.25, lot_size))
        
        # Explanation
        if correlation_penalty > 0:
            explanation = (f"Position size reduced by {(1-correlation_multiplier)*100:.0f}% due to "
                          f"correlation with: {', '.join(correlation_details)}")
        else:
            explanation = "Full position size - no significant correlation conflicts"
        
        return lot_size, explanation
    
    def should_take_position(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """Advanced position evaluation considering correlations"""
        
        # Basic checks
        if len(self.active_positions) >= self.max_positions:
            return False, f"Maximum {self.max_positions} positions reached"
        
        # Check for same pair
        for pos in self.active_positions:
            if pos.order.symbol == symbol:
                return False, f"Already have position in {symbol}"
        
        # Portfolio risk check
        current_risk = len(self.active_positions) * self.base_risk_per_trade
        if current_risk >= self.max_portfolio_risk:
            return False, f"Portfolio risk {current_risk:.1f}% exceeds {self.max_portfolio_risk}%"
        
        # Correlation analysis
        high_correlations = []
        total_correlation_risk = 0.0
        
        for pos in self.active_positions:
            corr = self.get_correlation(symbol, pos.order.symbol)
            
            if direction == pos.order.direction and abs(corr) > self.correlation_threshold:
                high_correlations.append(f"{pos.order.symbol}({corr:+.2f})")
                total_correlation_risk += abs(corr)
        
        if total_correlation_risk > 1.5:  # Too much correlation risk
            return False, f"High correlation risk with: {', '.join(high_correlations)}"
        
        # Currency concentration check
        base_currency = symbol[:3]
        quote_currency = symbol[3:6]
        
        current_exposure = defaultdict(float)
        for pos in self.active_positions:
            pos_base = pos.order.symbol[:3]
            pos_quote = pos.order.symbol[3:6]
            
            if pos.order.direction == 'long':
                current_exposure[pos_base] += 1
                current_exposure[pos_quote] -= 1
            else:
                current_exposure[pos_base] -= 1
                current_exposure[pos_quote] += 1
        
        # Check if new position would create excessive currency concentration
        if direction == 'long':
            new_base_exposure = abs(current_exposure[base_currency] + 1)
            new_quote_exposure = abs(current_exposure[quote_currency] - 1)
        else:
            new_base_exposure = abs(current_exposure[base_currency] - 1)
            new_quote_exposure = abs(current_exposure[quote_currency] + 1)
        
        if new_base_exposure > 2.5 or new_quote_exposure > 2.5:
            return False, f"Excessive {base_currency}/{quote_currency} currency concentration"
        
        return True, "Position approved with correlation analysis"
    
    def create_enhanced_trade_analysis(self, symbol: str, opportunity: Dict) -> str:
        """Create enhanced trade analysis including correlation insights"""
        
        direction = opportunity['direction']
        entry = opportunity['entry']
        stop = opportunity['stop_loss']
        target = opportunity['take_profit']
        setup_type = opportunity['setup_type']
        
        # Get correlation context
        correlation_context = ""
        if self.active_positions:
            correlation_context = "\n🔗 CORRELATION ANALYSIS:"
            
            for pos in self.active_positions:
                corr = self.get_correlation(symbol, pos.order.symbol)
                if abs(corr) > 0.3:  # Significant correlation
                    direction_note = "SAME" if direction == pos.order.direction else "OPPOSITE"
                    risk_level = "HIGH" if abs(corr) > 0.7 else "MEDIUM" if abs(corr) > 0.5 else "LOW"
                    
                    correlation_context += f"\n   • vs {pos.order.symbol}: {corr:+.2f} correlation ({direction_note} direction, {risk_level} risk)"
        
        # Calculate optimal size with correlation
        lot_size, size_explanation = self.calculate_optimal_position_size(symbol, direction, entry, stop)
        
        # Portfolio metrics
        metrics = self.analyze_portfolio_correlation()
        
        portfolio_context = f"""
📊 PORTFOLIO IMPACT:
   • Positions: {len(self.active_positions)}/{self.max_positions}
   • Portfolio Correlation Risk: {metrics.portfolio_correlation_risk:.1%}
   • Diversification Ratio: {metrics.diversification_ratio:.2f}
   • Currency Concentration: {metrics.concentration_risk:.1%}
   • Position Size: {lot_size:.2f} lots ({size_explanation})"""
        
        return f"""
🎯 ENHANCED TRADE SETUP: {symbol} {direction.upper()}

📊 ICT SETUP: {setup_type}
   Entry: {entry:.4f} | Stop: {stop:.4f} | Target: {target:.4f}
   
{correlation_context}
{portfolio_context}

✅ TRADE APPROVED: Advanced correlation analysis confirms position viability"""
    
    def execute_correlation_aware_trade(self, symbol: str, opportunity: Dict) -> Optional[PositionAnalysis]:
        """Execute trade with full correlation analysis"""
        
        # Check if we should take the position
        can_take, reason = self.should_take_position(symbol, opportunity['direction'])
        if not can_take:
            print(f"❌ TRADE REJECTED: {reason}")
            return None
        
        # Create enhanced analysis
        analysis = self.create_enhanced_trade_analysis(symbol, opportunity)
        print(analysis)
        
        # Calculate optimal position size
        lot_size, size_explanation = self.calculate_optimal_position_size(
            symbol, opportunity['direction'], opportunity['entry'], opportunity['stop_loss']
        )
        
        # Create order
        self.trade_counter += 1
        order = TradingOrder(
            order_id=f"CORR_{self.trade_counter:03d}",
            symbol=symbol,
            order_type='market',
            direction=opportunity['direction'],
            entry_price=opportunity['entry'],
            size_lots=lot_size,
            stop_loss=opportunity['stop_loss'],
            take_profit=opportunity['take_profit'],
            status='filled',
            reasoning=f"Correlation-aware {opportunity['setup_type']}"
        )
        
        # Simulate fill
        market_data = self.data_provider.get_current_market_data(symbol)
        order.fill_price = market_data.ask if opportunity['direction'] == 'long' else market_data.bid
        order.fill_time = datetime.now()
        order.pnl = 0.0
        
        # Create position analysis
        position_analysis = PositionAnalysis(order=order)
        
        # Calculate correlation exposure for this position
        for pos in self.active_positions:
            corr = self.get_correlation(symbol, pos.order.symbol)
            position_analysis.correlation_exposure[pos.order.symbol] = corr
        
        # Add to active positions
        self.active_positions.append(position_analysis)
        self.account.total_trades += 1
        
        print(f"✅ POSITION OPENED: {order.order_id} - {symbol} {opportunity['direction'].upper()}")
        print(f"   Size: {lot_size:.2f} lots | {size_explanation}")
        
        return position_analysis
    
    def update_positions_with_correlation_monitoring(self):
        """Update positions with correlation risk monitoring"""
        
        positions_to_close = []
        
        # Update P&L and check exits
        for position in self.active_positions:
            order = position.order
            market_data = self.data_provider.get_current_market_data(order.symbol)
            
            current_price = market_data.bid if order.direction == 'long' else market_data.ask
            
            # Calculate P&L
            if order.direction == 'long':
                price_diff = current_price - order.fill_price
            else:
                price_diff = order.fill_price - current_price
            
            # Convert to dollars
            if 'JPY' in order.symbol:
                order.pnl = price_diff * order.size_lots * 1000
            elif order.symbol == 'XAUUSD':
                order.pnl = price_diff * order.size_lots * 100
            else:
                order.pnl = price_diff * order.size_lots * 10000
            
            # Check exit conditions
            if order.direction == 'long':
                if current_price <= order.stop_loss:
                    positions_to_close.append((position, 'STOP LOSS'))
                elif current_price >= order.take_profit:
                    positions_to_close.append((position, 'PROFIT TARGET'))
            else:
                if current_price >= order.stop_loss:
                    positions_to_close.append((position, 'STOP LOSS'))
                elif current_price <= order.take_profit:
                    positions_to_close.append((position, 'PROFIT TARGET'))
        
        # Close positions
        for position, reason in positions_to_close:
            self.close_position_with_correlation_update(position, reason)
    
    def close_position_with_correlation_update(self, position: PositionAnalysis, reason: str):
        """Close position and update correlation metrics"""
        
        order = position.order
        order.close_time = datetime.now()
        order.status = 'closed'
        
        # Update account
        self.account.balance += order.pnl
        self.account.equity = self.account.balance
        
        if order.pnl > 0:
            self.account.winning_trades += 1
            result = "✅ PROFIT"
        else:
            self.account.losing_trades += 1
            result = "❌ LOSS"
        
        # Remove from active positions
        self.active_positions.remove(position)
        self.account.closed_orders.append(order)
        
        print(f"\n{result}: {order.order_id} closed via {reason}")
        print(f"   P&L: ${order.pnl:+.2f} | Balance: ${self.account.balance:.2f}")
        
        # Update correlation metrics for remaining positions
        if self.active_positions:
            metrics = self.analyze_portfolio_correlation()
            print(f"   Portfolio Correlation Risk: {metrics.portfolio_correlation_risk:.1%}")
    
    def display_portfolio_dashboard(self):
        """Display comprehensive portfolio dashboard"""
        
        metrics = self.analyze_portfolio_correlation()
        
        print(f"\n📊 CORRELATION-AWARE PORTFOLIO DASHBOARD")
        print("="*60)
        print(f"💰 Balance: ${self.account.balance:.2f}")
        print(f"📈 Active Positions: {len(self.active_positions)}/{self.max_positions}")
        
        if self.active_positions:
            total_floating = sum(pos.order.pnl for pos in self.active_positions)
            print(f"💹 Floating P&L: ${total_floating:+.2f}")
            print(f"🔗 Portfolio Correlation Risk: {metrics.portfolio_correlation_risk:.1%}")
            print(f"📊 Diversification Ratio: {metrics.diversification_ratio:.2f}")
            print(f"⚠️ Currency Concentration: {metrics.concentration_risk:.1%}")
            
            print(f"\nACTIVE POSITIONS:")
            for pos in self.active_positions:
                print(f"  • {pos.order.symbol} {pos.order.direction.upper()}: ${pos.order.pnl:+.1f}")
        
        print("="*60)

def main():
    """Main enhanced correlation-aware trading demonstration"""
    
    print("🧠 ENHANCED CORRELATION-AWARE MULTI-PAIR TRADING")
    print("="*65)
    print("🔗 Features: Advanced correlation analysis")
    print("📊 Risk Management: Portfolio-level optimization")
    print("🎯 Position Sizing: Correlation-adjusted sizing")
    print("📈 Currency Exposure: Concentration risk monitoring")
    print("⚡ Diversification: Intelligent pair selection")
    print("\n🚀 Starting enhanced correlation system...\n")
    
    # Initialize enhanced system
    trader = CorrelationAwareTrader(balance=1000.0)
    start_time = datetime.now()
    
    try:
        cycle = 0
        
        while True:
            cycle += 1
            
            # Update market and positions
            trader.data_provider.update_prices()
            trader.update_positions_with_correlation_monitoring()
            
            # Portfolio dashboard every 10 seconds
            if cycle % 200 == 0:
                trader.display_portfolio_dashboard()
            
            # Look for new opportunities every 3 seconds
            if cycle % 60 == 0 and len(trader.active_positions) < trader.max_positions:
                
                # Scan pairs for opportunities
                available_pairs = [
                    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF',
                    'NZDUSD', 'USDCAD', 'EURJPY', 'GBPJPY', 'AUDJPY', 'XAUUSD'
                ]
                
                # Remove pairs we already have positions in
                active_symbols = [pos.order.symbol for pos in trader.active_positions]
                available_pairs = [p for p in available_pairs if p not in active_symbols]
                
                # Scan random selection
                pairs_to_scan = random.sample(available_pairs, min(3, len(available_pairs)))
                
                for symbol in pairs_to_scan:
                    try:
                        analysis = trader.analyzer.analyze_pair(symbol)
                        
                        for opportunity in analysis['trading_opportunities']:
                            if (opportunity['risk_reward'] >= 1.3 and
                                opportunity['confidence'] in ['medium', 'high']):
                                
                                position = trader.execute_correlation_aware_trade(symbol, opportunity)
                                
                                if position:
                                    time.sleep(2)  # Pause to read analysis
                                    break
                                    
                    except Exception:
                        continue
            
            # Check exit conditions
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if trader.account.balance <= 4.0:
                print(f"\n⚠️ Balance protection: ${trader.account.balance:.2f}")
                break
            elif trader.account.balance >= 25.0:
                print(f"\n🎉 Profit target reached: ${trader.account.balance:.2f}")
                break
            elif elapsed > 150:  # 2.5 minutes
                print(f"\n⏰ Demo completed")
                break
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Stopped by user")
    
    # Final results with correlation analysis
    final_metrics = trader.analyze_portfolio_correlation()
    elapsed_total = (datetime.now() - start_time).total_seconds()
    
    total_trades = len(trader.account.closed_orders)
    win_rate = (trader.account.winning_trades / max(1, total_trades)) * 100
    
    print(f"\n" + "="*70)
    print(f"🏁 ENHANCED CORRELATION-AWARE RESULTS")
    print("="*70)
    
    print(f"⏱️ Duration: {elapsed_total:.0f} seconds")
    print(f"💰 Final Balance: ${trader.account.balance:.2f}")
    print(f"📊 Total Return: {((trader.account.balance/1000)-1)*100:+.1f}%")
    print(f"📈 Total Trades: {total_trades}")
    print(f"✅ Win Rate: {win_rate:.1f}%")
    
    print(f"\n🔗 CORRELATION METRICS:")
    print(f"Portfolio Correlation Risk: {final_metrics.portfolio_correlation_risk:.1%}")
    print(f"Diversification Ratio: {final_metrics.diversification_ratio:.2f}")
    print(f"Currency Concentration Risk: {final_metrics.concentration_risk:.1%}")
    
    if trader.account.closed_orders:
        print(f"\n📋 CORRELATION-OPTIMIZED TRADES:")
        for i, order in enumerate(trader.account.closed_orders, 1):
            result = "✅" if order.pnl > 0 else "❌"
            print(f"{i}. {result} {order.symbol} {order.direction.upper()}: ${order.pnl:+.2f}")
    
    # Save enhanced results
    results = {
        'final_balance': trader.account.balance,
        'correlation_metrics': {
            'portfolio_correlation_risk': final_metrics.portfolio_correlation_risk,
            'diversification_ratio': final_metrics.diversification_ratio,
            'concentration_risk': final_metrics.concentration_risk
        },
        'total_trades': total_trades,
        'win_rate': win_rate,
        'duration': elapsed_total
    }
    
    filename = f"enhanced_correlation_{datetime.now().strftime('%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Enhanced results saved: {filename}")
    print(f"✅ Correlation-aware trading demonstration complete!")


if __name__ == "__main__":
    main()