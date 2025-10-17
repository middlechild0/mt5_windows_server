#!/usr/bin/env python3
"""
AI Trading Integration Framework
===============================
This script integrates the ICT Knowledge AI system with the trading backtest engine
to create a complete AI-driven trading system that can:
1. Query ICT knowledge for trading decisions
2. Generate trading signals based on learned concepts
3. Execute backtests with AI-powered signal generation
4. Provide explainable trading decisions
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional
from ict_knowledge_ai import ICTKnowledgeAI
from trading_backtest import AITradingBacktest, PerformanceMetrics

class ICTTradingAI:
    """
    Advanced AI trading system that combines ICT knowledge with technical analysis
    to generate trading signals with explanations.
    """
    
    def __init__(self, knowledge_ai: ICTKnowledgeAI, confidence_threshold: float = 0.3):
        """
        Initialize the ICT Trading AI.
        
        Args:
            knowledge_ai: Trained ICT Knowledge AI system
            confidence_threshold: Minimum confidence score for signal generation
        """
        self.knowledge_ai = knowledge_ai
        self.confidence_threshold = confidence_threshold
        
        # Market context keywords for different scenarios
        self.market_scenarios = {
            'trend_up': ['bullish', 'uptrend', 'higher highs', 'buy', 'long', 'bullish order block'],
            'trend_down': ['bearish', 'downtrend', 'lower lows', 'sell', 'short', 'bearish order block'],
            'consolidation': ['range', 'sideways', 'consolidation', 'chop', 'no trade'],
            'reversal': ['reversal', 'change of character', 'break of structure', 'shift'],
            'entry': ['entry', 'enter', 'position', 'trade setup'],
            'exit': ['exit', 'take profit', 'stop loss', 'close position'],
            'risk': ['risk management', 'position sizing', 'stop loss', 'drawdown']
        }
        
        # Signal generation rules based on ICT concepts
        self.ict_rules = {
            'order_blocks': ['order block', 'institutional level', 'supply', 'demand'],
            'liquidity': ['liquidity grab', 'stop hunt', 'sweep', 'raid'],
            'structure': ['market structure', 'break of structure', 'change of character'],
            'time': ['london session', 'new york session', 'asian session', 'kill zone'],
            'patterns': ['fair value gap', 'imbalance', 'inefficiency', 'displacement']
        }

    def analyze_market_context(self, candle_data: pd.Series) -> Dict:
        """
        Analyze current market context using technical indicators.
        
        Args:
            candle_data: Current candle data with indicators
            
        Returns:
            dict: Market context analysis
        """
        context = {}
        
        # Price action analysis
        close = candle_data['Close']
        high = candle_data['High']
        low = candle_data['Low']
        open_price = candle_data['Open']
        
        # Trend analysis using SMAs
        if 'SMA_20' in candle_data and 'SMA_50' in candle_data:
            sma_20 = candle_data['SMA_20']
            sma_50 = candle_data['SMA_50']
            
            if pd.notna(sma_20) and pd.notna(sma_50):
                if sma_20 > sma_50:
                    context['trend'] = 'bullish'
                elif sma_20 < sma_50:
                    context['trend'] = 'bearish'
                else:
                    context['trend'] = 'neutral'
            else:
                context['trend'] = 'neutral'
        else:
            context['trend'] = 'neutral'
        
        # Volatility analysis
        if 'ATR' in candle_data and pd.notna(candle_data['ATR']):
            atr = candle_data['ATR']
            volatility_pct = atr / close
            if volatility_pct > 0.002:
                context['volatility'] = 'high'
            elif volatility_pct > 0.001:
                context['volatility'] = 'medium'
            else:
                context['volatility'] = 'low'
        else:
            context['volatility'] = 'medium'
        
        # RSI analysis
        if 'RSI' in candle_data and pd.notna(candle_data['RSI']):
            rsi = candle_data['RSI']
            if rsi > 70:
                context['momentum'] = 'overbought'
            elif rsi < 30:
                context['momentum'] = 'oversold'
            else:
                context['momentum'] = 'neutral'
        else:
            context['momentum'] = 'neutral'
        
        # Candle pattern
        body_size = abs(close - open_price)
        total_range = high - low
        
        if total_range > 0:
            body_ratio = body_size / total_range
            if body_ratio > 0.7:
                context['candle_type'] = 'strong_momentum'
            elif body_ratio < 0.3:
                context['candle_type'] = 'doji_indecision'
            else:
                context['candle_type'] = 'normal'
        else:
            context['candle_type'] = 'normal'
        
        # Price position relative to range
        if total_range > 0:
            close_position = (close - low) / total_range
            if close_position > 0.7:
                context['price_position'] = 'near_high'
            elif close_position < 0.3:
                context['price_position'] = 'near_low'
            else:
                context['price_position'] = 'middle'
        else:
            context['price_position'] = 'middle'
        
        return context

    def generate_market_query(self, context: Dict) -> str:
        """
        Generate a query for the ICT knowledge AI based on market context.
        
        Args:
            context: Market context analysis
            
        Returns:
            str: Query string for knowledge AI
        """
        # Build query based on market conditions
        query_parts = []
        
        # Add trend context
        if context['trend'] == 'bullish':
            query_parts.append("bullish market uptrend buy long entry")
        elif context['trend'] == 'bearish':
            query_parts.append("bearish market downtrend sell short entry")
        else:
            query_parts.append("sideways market range trading")
        
        # Add momentum context
        if context['momentum'] == 'overbought':
            query_parts.append("overbought levels reversal sell")
        elif context['momentum'] == 'oversold':
            query_parts.append("oversold levels reversal buy")
        
        # Add volatility context
        if context['volatility'] == 'high':
            query_parts.append("high volatility breakout")
        elif context['volatility'] == 'low':
            query_parts.append("low volatility consolidation")
        
        # Add candle pattern context
        if context['candle_type'] == 'strong_momentum':
            query_parts.append("strong momentum continuation")
        elif context['candle_type'] == 'doji_indecision':
            query_parts.append("indecision reversal pattern")
        
        # Add price position context
        if context['price_position'] == 'near_high':
            query_parts.append("resistance levels supply zone")
        elif context['price_position'] == 'near_low':
            query_parts.append("support levels demand zone")
        
        # Always include general trading concepts
        query_parts.append("entry setup risk management")
        
        return " ".join(query_parts)

    def interpret_ict_response(self, results: List[Dict], context: Dict) -> Dict:
        """
        Interpret ICT knowledge AI response to generate trading decision.
        
        Args:
            results: Results from knowledge AI query
            context: Market context
            
        Returns:
            dict: Trading decision with reasoning
        """
        if not results:
            return {'signal': 'hold', 'confidence': 0, 'reasoning': 'No relevant ICT concepts found'}
        
        # Analyze the content for trading signals
        combined_text = " ".join([r['chunk'] for r in results]).lower()
        
        # Calculate signal scores
        buy_score = 0
        sell_score = 0
        hold_score = 0
        
        # Check for bullish indicators
        bullish_keywords = ['buy', 'long', 'bullish', 'demand', 'support', 'bounce', 'higher', 'up']
        for keyword in bullish_keywords:
            buy_score += combined_text.count(keyword)
        
        # Check for bearish indicators
        bearish_keywords = ['sell', 'short', 'bearish', 'supply', 'resistance', 'drop', 'lower', 'down']
        for keyword in bearish_keywords:
            sell_score += keyword in combined_text
        
        # Check for hold indicators
        hold_keywords = ['wait', 'hold', 'no trade', 'avoid', 'caution', 'uncertain']
        for keyword in hold_keywords:
            hold_score += combined_text.count(keyword)
        
        # Adjust scores based on market context
        if context['trend'] == 'bullish':
            buy_score *= 1.5
        elif context['trend'] == 'bearish':
            sell_score *= 1.5
        
        if context['momentum'] == 'overbought':
            sell_score *= 1.3
        elif context['momentum'] == 'oversold':
            buy_score *= 1.3
        
        # Determine signal
        total_score = buy_score + sell_score + hold_score
        if total_score == 0:
            return {'signal': 'hold', 'confidence': 0, 'reasoning': 'No clear direction from ICT analysis'}
        
        # Calculate confidence based on highest scoring result similarity
        max_similarity = max([r['similarity'] for r in results])
        base_confidence = max_similarity
        
        if buy_score > sell_score and buy_score > hold_score:
            signal = 'buy'
            confidence = base_confidence * (buy_score / total_score)
        elif sell_score > buy_score and sell_score > hold_score:
            signal = 'sell'
            confidence = base_confidence * (sell_score / total_score)
        else:
            signal = 'hold'
            confidence = base_confidence * (hold_score / total_score)
        
        # Generate reasoning
        top_result = results[0]
        reasoning = f"ICT Analysis: {signal.upper()} signal with {confidence:.2f} confidence. "
        reasoning += f"Based on: {top_result['chunk'][:100]}... "
        reasoning += f"Market context: {context['trend']} trend, {context['momentum']} momentum."
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'ict_concepts': [r['chunk'][:200] for r in results[:2]]  # Top 2 concepts
        }

    def generate_signal(self, candle_data: pd.Series) -> str:
        """
        Generate trading signal using ICT knowledge and technical analysis.
        
        Args:
            candle_data: Current candle data
            
        Returns:
            str: 'buy', 'sell', or 'hold'
        """
        # Analyze market context
        context = self.analyze_market_context(candle_data)
        
        # Generate query for ICT knowledge
        query = self.generate_market_query(context)
        
        # Query the knowledge AI
        results = self.knowledge_ai.query(query, top_k=3)
        
        # Interpret the response
        decision = self.interpret_ict_response(results, context)
        
        # Apply confidence threshold
        if decision['confidence'] < self.confidence_threshold:
            return 'hold'
        
        # Store decision for later analysis (optional)
        self.last_decision = decision
        
        return decision['signal']

    def get_last_decision_explanation(self) -> str:
        """Get explanation for the last trading decision."""
        if hasattr(self, 'last_decision'):
            return self.last_decision['reasoning']
        return "No recent decision available."


class IntegratedTradingSystem:
    """
    Complete integrated trading system combining ICT knowledge with backtesting.
    """
    
    def __init__(self, 
                 transcript_file: str = "playlist_transcripts.txt",
                 initial_balance: float = 10000,
                 confidence_threshold: float = 0.3):
        """
        Initialize the integrated trading system.
        
        Args:
            transcript_file: Path to ICT transcript file
            initial_balance: Starting balance for backtesting
            confidence_threshold: Minimum confidence for trade signals
        """
        self.transcript_file = transcript_file
        self.initial_balance = initial_balance
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.knowledge_ai = None
        self.trading_ai = None
        self.backtester = None
        
    def setup_system(self) -> bool:
        """
        Set up the complete trading system.
        
        Returns:
            bool: True if setup successful
        """
        print("🚀 Setting up Integrated AI Trading System...")
        
        # Initialize and train knowledge AI
        print("📚 Loading ICT Knowledge AI...")
        self.knowledge_ai = ICTKnowledgeAI(max_features=5000, n_clusters=10, chunk_size=250)
        
        # Try to load existing model or train new one
        if not self.knowledge_ai.load_model():
            print("🎓 Training new ICT Knowledge AI...")
            if not self.knowledge_ai.train(self.transcript_file):
                print("❌ Failed to train knowledge AI. Check transcript file.")
                return False
            self.knowledge_ai.save_model()
        
        # Initialize trading AI
        print("🤖 Setting up Trading AI...")
        self.trading_ai = ICTTradingAI(self.knowledge_ai, self.confidence_threshold)
        
        # Initialize backtester with ICT AI signal function
        print("📊 Setting up Backtester...")
        self.backtester = AITradingBacktest(
            initial_balance=self.initial_balance,
            position_size=1000,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            breakeven_trigger_pct=0.01
        )
        
        # Replace the backtester's AI signal function with our ICT-based one
        self.backtester.ai_signal = self.trading_ai.generate_signal
        
        print("✅ System setup complete!")
        return True
    
    def run_backtest(self, 
                     symbol: str = "EURUSD=X",
                     period: str = "6mo", 
                     interval: str = "1h") -> Optional[PerformanceMetrics]:
        """
        Run a complete backtest using ICT knowledge-based signals.
        
        Args:
            symbol: Trading symbol
            period: Time period for data
            interval: Data interval
            
        Returns:
            PerformanceMetrics: Backtest results
        """
        if not self.backtester:
            print("❌ System not set up. Run setup_system() first.")
            return None
        
        print(f"📈 Running ICT AI backtest on {symbol}...")
        
        # Load data
        data = self.backtester.load_historical_data(symbol, period, interval)
        if data.empty:
            print("❌ No data available for backtesting.")
            return None
        
        # Run backtest
        metrics = self.backtester.run_backtest(data)
        
        # Display results
        self.backtester.print_trade_log(limit=15)
        self.backtester.plot_results(metrics)
        
        return metrics
    
    def interactive_trading_session(self):
        """Run an interactive session for testing trading decisions."""
        if not self.trading_ai:
            print("❌ System not set up. Run setup_system() first.")
            return
        
        print("\n" + "="*60)
        print("🎯 ICT AI Trading Decision Tester")
        print("="*60)
        print("Enter market conditions to get AI trading decisions.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                print("\n📊 Enter current market conditions:")
                close = float(input("Close price: "))
                high = float(input("High price: "))
                low = float(input("Low price: "))
                
                # Create mock candle data
                candle = pd.Series({
                    'Close': close,
                    'High': high,
                    'Low': low,
                    'Open': close,  # Simplified
                    'SMA_20': close * 1.001,  # Mock SMA
                    'SMA_50': close * 0.999,  # Mock SMA
                    'RSI': 50,  # Neutral RSI
                    'ATR': (high - low)  # Simple ATR
                })
                
                # Get AI decision
                signal = self.trading_ai.generate_signal(candle)
                explanation = self.trading_ai.get_last_decision_explanation()
                
                print(f"\n🤖 AI Decision: {signal.upper()}")
                print(f"💡 Reasoning: {explanation}")
                
                cont = input("\nTest another scenario? (y/n): ").strip().lower()
                if cont != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Session ended by user.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to demonstrate the integrated trading system."""
    print("🎯 ICT AI Trading Integration Framework")
    print("=" * 60)
    
    # Create integrated system
    system = IntegratedTradingSystem(
        transcript_file="playlist_transcripts.txt",
        initial_balance=10000,
        confidence_threshold=0.25
    )
    
    # Setup the system
    if not system.setup_system():
        print("❌ System setup failed. Exiting...")
        return
    
    # Menu system
    while True:
        print("\n🎮 Choose an option:")
        print("1. Run ICT AI Backtest")
        print("2. Test Trading Decisions Interactively") 
        print("3. Query ICT Knowledge Directly")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\n📊 Starting ICT AI Backtest...")
            system.run_backtest()
            
        elif choice == '2':
            system.interactive_trading_session()
            
        elif choice == '3':
            # Direct knowledge query
            while True:
                question = input("\n❓ Ask ICT AI: ").strip()
                if question.lower() in ['quit', 'exit', '']:
                    break
                
                results = system.knowledge_ai.query(question)
                if results:
                    print(f"\n📚 ICT Knowledge Response:")
                    for i, result in enumerate(results[:2], 1):
                        print(f"\n{i}. {result['chunk'][:300]}...")
                else:
                    print("❌ No relevant information found.")
                    
        elif choice == '4':
            print("👋 Thank you for using ICT AI Trading System!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()